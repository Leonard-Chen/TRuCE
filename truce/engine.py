#!/usr/bin/env python3
import copy
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import build_relation_vocab
from .interventions import make_observation, make_training_observation
from .metrics import (build_train_facts, evaluate_relations, predictions_from_scores,
                      type_correction_metrics)
from .model import DocREModel
from .losses import compute_loss
from .runtime import (choose_device, encode_split, environment_info, load_split,
                      read_json, resolve_path, seed_everything, sha256, tensorize, write_json, source_manifest, resolve_resume_path, relocate_config)
from .tokenization import create_tokenizer


TEACHER_VARIANTS = {'B4', 'B5', 'B6', 'B7', 'B8'}


class TeacherStore:
    def __init__(self, config, required=False):
        self.config = config
        self.directory = resolve_path(config['teacher']['cache_dir'])
        self.cache = {}
        self.confusion = None
        self.metadata = None

        manifest = self.directory / 'manifest.json'

        if not manifest.exists() and (required or config['teacher'].get('expected_manifest_sha256')):
            raise FileNotFoundError('缺少训练时记录的类型器 manifest：' + str(manifest))

        if manifest.exists():
            self.metadata = read_json(manifest)
            expected_hash = config['teacher'].get('expected_manifest_sha256')
            if expected_hash and sha256(manifest) != expected_hash:
                raise ValueError('类型器已变化，与训练 checkpoint 中记录的版本不一致。')
            provenance = self.metadata.get('provenance', {})
            for key, actual in (('encoder_name', config['model'].get('encoder_name')),
                                ('encoder_backend', config['model'].get('encoder_backend', 'hf')),
                                ('encoder_revision_requested', config['model'].get('encoder_revision', 'main')),
                                ('max_length', config['data'].get('max_length', 512)),
                                ('overlap', config['data'].get('overlap', 128))):
                if provenance.get(key) != actual:
                    raise ValueError('类型器配置与当前实验不匹配：' + key)

        confusion = self.directory / 'confusion.json'
        if confusion.exists():
            self._verify_file(confusion)
            value = read_json(confusion)
            if not isinstance(value, dict) or value.get('source') != 'train_q_local_oof':
                raise ValueError('混淆矩阵缺少 train_q_local_oof 来源证明。')
            if value.get('source_sha256') != sha256(config['data']['train']):
                raise ValueError('混淆矩阵不是当前训练集生成，请重新准备类型缓存。')
            self.confusion = value.get('matrix', value) if isinstance(value, dict) else value

        if required and not self.directory.exists():
            raise FileNotFoundError('请先运行 scripts/prepare_teachers.py 生成真实的 OOF 类型预测。')

    def _verify_file(self, path):
        if self.metadata is not None:
            expected = self.metadata.get('files', {}).get(path.name)
            if not expected or sha256(path) != expected:
                raise ValueError('类型缓存文件 SHA 校验失败：' + str(path))

    def get(self, split, doc_id):
        if split not in self.cache:
            path = self.directory / (split + '.json')
            if not path.exists():
                raise FileNotFoundError('缺少类型预测文件：' + str(path))

            self._verify_file(path)
            value = read_json(path)
            expected = sha256(self.config['data'][split])
            if value.get('source_sha256') != expected:
                raise ValueError('类型缓存数据版本不匹配，请重新生成：' + str(path))
            if split == 'train' and value.get('inference_mode') != 'oof':
                raise ValueError('训练文档只能读取 OOF 类型预测，禁止 in-sample 类型标签泄漏。')

            self.cache[split] = value.get('docs', value.get('documents', value))

        if doc_id not in self.cache[split]:
            raise KeyError('类型缓存缺少文档：' + doc_id)

        return self.cache[split][doc_id]


def prepare(config, split, vocab=None, limit=None):
    docs = load_split(config['data'][split], limit)
    if vocab is None:
        vocab = build_relation_vocab(load_split(config['data']['train']))
    tokenizer = create_tokenizer(config)
    encoded = encode_split(docs, tokenizer, vocab, config)
    return docs, encoded, vocab


def observation_for(doc, condition, rate, noise_seed, store, split):
    local = store.get(split, doc['doc_id'])['q_local'] if condition == 'local' else None
    if condition == 'confusion' and store.confusion is None:
        raise ValueError('混淆干预必须使用 train OOF 混淆矩阵；请先准备类型器。')
    return make_observation(doc, condition=condition, rate=rate, seed=noise_seed,
                            confusion=store.confusion, local_probs=local)


def training_observation(doc, config, epoch, store):
    seed = int(config['training']['seed'])
    if config['training'].get('input_strategy', 'mixed') == 'clean':
        return observation_for(doc, 'clean', 0.0, seed, store, 'train')
    return make_training_observation(doc, epoch, seed=seed, confusion=store.confusion)


@torch.no_grad()
def evaluate_model(model, documents, encoded, vocab, config, store, split, device,
                   condition='clean', rate=0.0, noise_seed=101, train_facts=None):
    '''
    模型评估
    '''
    model.eval()

    predictions, per_document, manifests = [], [], []
    references, observations, corrected, consistent, teacher_correct = [], [], [], [], []
    gate_sum = residual_sum = entity_count = 0
    evidence_tp = evidence_pred = evidence_gold = evidence_pairs = 0
    start = time.perf_counter()

    for doc, item in zip(documents, encoded):
        obs = observation_for(doc, condition, rate, noise_seed, store, split)
        q = store.get(split, doc['doc_id'])['q_doc'] if model.variant in TEACHER_VARIANTS else None
        inputs, targets = tensorize(item, obs, q, device)
        output = model(inputs)
        probabilities = output['logits'].sigmoid().cpu().numpy()
        doc_predictions = predictions_from_scores(doc, probabilities, vocab,
                                                   config['evaluation'].get('threshold', 0.5))
        predictions.extend(doc_predictions)
        per_document.append({'doc_id': doc['doc_id'], 'predictions': doc_predictions})
        manifests.append(obs)
        references.extend(doc['reference_types'])
        consistent.extend(doc['class_consistent'])
        observations.extend(obs['observed_types'])
        corrected.extend(output['corrected_probs'].cpu().tolist())
        teacher_correct.extend([int(np.argmax(prob)) == ref for prob, ref in zip(q, doc['reference_types'])]
                               if q is not None else [None] * len(doc['reference_types']))
        entity_count += len(obs['observed_types'])
        gate_sum += float(output['gates'].sum())
        residual_sum += float(output['type_logits'].norm())

        for pair_index in torch.where(targets['evidence_mask'])[0].tolist():
            evidence_pairs += 1
            truth = set(torch.where(targets['evidence_labels'][pair_index] > 0)[0].tolist())
            count = min(2, output['evidence_logits'].shape[-1])
            predicted = set(output['evidence_logits'][pair_index].topk(count).indices.tolist())
            evidence_tp += len(truth & predicted)
            evidence_pred += len(predicted)
            evidence_gold += len(truth)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    seconds = time.perf_counter() - start
    metrics = evaluate_relations(documents, predictions, train_facts)
    metrics['type'] = type_correction_metrics(references, observations, corrected, consistent)
    metrics['type']['interpretation'] = '无外部类型通道；不主张类型纠正' if model.variant == 'B0' else (
        '显式软纠正' if model.uses_c else '全文类型替换' if model.variant == 'B4' else '无显式纠正模块')

    if model.requires_teacher:
        metrics['type']['teacher_groups'] = {}
        for status in (True, False):
            selected = [i for i, correct in enumerate(teacher_correct) if correct == status]
            metrics['type']['teacher_groups']['teacher_correct' if status else 'teacher_wrong'] = type_correction_metrics(
                [references[i] for i in selected], [observations[i] for i in selected],
                [corrected[i] for i in selected], [consistent[i] for i in selected])

    metrics['evidence'] = {'labeled_pairs': evidence_pairs,
                           'candidate_pairs': sum(len(x['pairs']) for x in encoded),
                           'top_k': 2, 'precision': evidence_tp / max(evidence_pred, 1),
                           'recall': evidence_tp / max(evidence_gold, 1),
                           'f1': 2 * evidence_tp / max(evidence_pred + evidence_gold, 1),
                           'scope': '仅非空证据标注实体对；参考标注可能不完备'}

    metrics['diagnostics'] = {'mean_gate': gate_sum / max(entity_count, 1),
                              'mean_type_residual_norm': residual_sum / max(len(documents), 1),
                              'seconds': seconds, 'teacher_cached': True,
                              'timing_scope': '缓存类型预测的逐文档评估，含指标整理，不是冷启动模型延时'}

    return {'metrics': metrics, 'predictions': predictions, 'documents': per_document,
            'observations': manifests, 'condition': condition, 'rate': rate,
            'noise_seed': noise_seed}


def _save_checkpoint(model, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(directory / 'model.safetensors'))
    write_json(directory / 'checkpoint.json', metadata)


def train(config, output=None, device='auto', resume=None):
    if config['model']['variant'] == 'E1':
        from .etmier_engine import train as train_etmier
        return train_etmier(config, output, device, resume=resume)

    config = copy.deepcopy(config)
    training = config['training']
    if int(training['epochs']) < 1 or (training.get('max_steps') is not None and int(training['max_steps']) < 1):
        raise ValueError('epochs 和指定的 max_steps 必须为正整数。')

    seed = int(training['seed'])
    seed_everything(seed, training.get('threads', 2))
    device = choose_device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    if resume:
        run_dir, _ = resolve_resume_path(resume)
        if output and resolve_path(output).resolve() != run_dir.resolve():
            raise ValueError('--resume 必须使用原运行目录；迁移时复制整个运行目录。')
    else:
        run_dir = resolve_path(output) if output else resolve_path(config['output']['directory']) / (
            config['model']['variant'] + '_seed' + str(seed) + '_' + time.strftime('%Y%m%d_%H%M%S'))
        if (run_dir / 'run.json').exists():
            raise FileExistsError('运行目录已经存在，请用 --resume 续训或更换 --output。')

    run_dir.mkdir(parents=True, exist_ok=True)
    train_docs, train_encoded, vocab = prepare(config, 'train', limit=training.get('limit_train'))
    dev_docs, dev_encoded, _ = prepare(config, 'dev', vocab, training.get('limit_dev'))
    if not train_docs or not dev_docs:
        raise ValueError('训练集与开发集均不能为空。')

    store = TeacherStore(config, config['model']['variant'] in TEACHER_VARIANTS)
    if store.confusion is None:
        raise ValueError('统一开发选模需要 C3 混淆矩阵，请先准备 OOF 类型缓存。')
    store.get('train', train_docs[0]['doc_id'])
    if store.metadata is None:
        raise ValueError('正式流水线必须有 teacher manifest，不能只提供无来源的概率文件。')

    config['teacher']['expected_manifest_sha256'] = sha256(store.directory / 'manifest.json')
    model = DocREModel(config, len(vocab)).to(device)
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder_params if name.startswith('encoder.') else head_params).append(parameter)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': training['learning_rate']},
        {'params': head_params, 'lr': training.get('head_learning_rate', training['learning_rate'])},
    ], weight_decay=training.get('weight_decay', 0.01))

    facts = build_train_facts(load_split(config['data']['train']))

    metadata = {'config': config, 'relation_vocab': vocab, 'environment': environment_info(),
                'data_sha256': {key: sha256(config['data'][key]) for key in ('train', 'dev', 'test')},
                'teacher_manifest_sha256': sha256(store.directory / 'manifest.json')
                    if (store.directory / 'manifest.json').exists() else None,
                'source_files_sha256': source_manifest(),
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'microbatch_documents': 1, 'effective_batch_documents': training['accumulation_steps'],
                'status': 'running', 'result_scope': 'smoke_only' if config['model'].get('encoder_backend') == 'tiny' else 'research'}

    from .training import run_training

    def training_step(index, epoch):
        doc, encoded = train_docs[index], train_encoded[index]
        observation = training_observation(doc, config, epoch, store)
        q = store.get('train', doc['doc_id'])['q_doc'] if model.variant in TEACHER_VARIANTS else None
        inputs, targets = tensorize(encoded, observation, q, device)
        losses = compute_loss(model(inputs), targets, config.get('loss', {}))
        return losses, {'doc_id': doc['doc_id'], 'observation': observation}

    def evaluate_epoch():
        clean = evaluate_model(model, dev_docs, dev_encoded, vocab, config, store, 'dev', device,
                               train_facts=facts)
        noisy = evaluate_model(model, dev_docs, dev_encoded, vocab, config, store, 'dev', device,
                               'confusion', 0.2, config['evaluation'].get('selection_noise_seed', 101), facts)
        return clean, noisy

    return run_training(model, optimizer, train_docs, config, run_dir, metadata,
                        training_step, evaluate_epoch, resume=resume)



def checkpoint_variant(path):
    path = Path(path).resolve()
    path = path / 'best' if (path / 'best').exists() else path
    return read_json(path / 'checkpoint.json')['config']['model']['variant']


def load_checkpoint(path, device='auto'):
    path = Path(path).resolve()
    if (path / 'best').exists():
        path = path / 'best'

    metadata = read_json(path / 'checkpoint.json')
    config, vocab = relocate_config(metadata['config']), metadata['relation_vocab']
    if config['model']['variant'] == 'E1':
        from .etmier_engine import load_checkpoint as load_etmier
        return load_etmier(path, device)

    for split, expected in metadata['data_sha256'].items():
        if sha256(config['data'][split]) != expected:
            raise ValueError('当前数据与训练时版本不一致：' + split)

    if metadata.get('teacher_manifest_sha256'):
        config['teacher']['expected_manifest_sha256'] = metadata['teacher_manifest_sha256']

    seed_everything(config['training']['seed'], config['training'].get('threads', 2))
    device = choose_device(device)
    model = DocREModel(config, len(vocab)).to(device)
    model.load_state_dict(load_file(str(path / 'model.safetensors'), device=str(device)))
    model.eval()
    return model, config, vocab, device


def evaluate(checkpoint, split='test', condition='clean', rate=0.0, noise_seed=101,
             output=None, device='auto', limit=None):

    if checkpoint_variant(checkpoint) == 'E1':
        from .etmier_engine import evaluate as evaluate_etmier
        return evaluate_etmier(checkpoint, split, condition, rate, noise_seed, output, device, limit)

    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs, encoded, _ = prepare(config, split, vocab, limit)
    store = TeacherStore(config, model.variant in TEACHER_VARIANTS)
    facts = build_train_facts(load_split(config['data']['train']))

    result = evaluate_model(model, docs, encoded, vocab, config, store, split, device,
                            condition, rate, noise_seed, facts)

    result.update({'split': split, 'checkpoint': str(Path(checkpoint).resolve()),
                   'doc_ids': [d['doc_id'] for d in docs],
                   'training_seed': config['training']['seed'],
                   'variant': model.variant,
                   'source_sha256': sha256(config['data'][split]),
                   'threshold': config['evaluation'].get('threshold', 0.5)})

    if output:
        write_json(output, result)
        write_json(Path(output).with_suffix('.predictions.json'), result['predictions'])

    return result
