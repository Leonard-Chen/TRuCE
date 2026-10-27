#!/usr/bin/env python3
'''ET-MIER 作者模型的统一实验协议适配，不含推理阶段双模型融合'''

import copy
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import build_relation_vocab
from .engine import TeacherStore, observation_for, training_observation
from .etmier_data import encode_etmier_document, collate_etmier
from .etmier_model import ETMIERAdapter
from .metrics import build_train_facts, evaluate_relations
from .runtime import (choose_device, environment_info, load_split, read_json, resolve_path,
                      seed_everything, sha256, source_manifest, write_json,
                      resolve_resume_path, relocate_config)
from .tokenization import create_tokenizer


TARGET_FIELDS = {'labels', 'sent_labels', 'reference_epair_types', 'reference_epair_mask', 'evidence_mask'}


def _batch(doc, tokenizer, vocab, observation, config, device):
    feature = encode_etmier_document(doc, tokenizer, vocab, observation,
                                     max_seq_length=config.get('etmier', {}).get('max_seq_length', 1022),
                                     transformer_type=config.get('etmier', {}).get('transformer_type', 'bert'))
    packed = collate_etmier([feature], device=device, pad_token_id=tokenizer.pad_token_id)
    targets = {key: packed[key] for key in TARGET_FIELDS if key in packed}
    inputs = {key: value for key, value in packed.items() if key not in TARGET_FIELDS}
    return feature, inputs, targets


def _official_predictions(doc, feature, matrix, vocab):
    inverse = {index + 1: relation for relation, index in vocab.items()}
    predictions = []
    for (head, tail), row in zip(feature['hts'], matrix.detach().cpu().tolist()):
        for index, selected in enumerate(row):
            if index and selected:
                predictions.append({'title': doc['title'], 'h_idx': head, 't_idx': tail,
                                    'r': inverse[index]})
    return predictions


@torch.no_grad()
def _evaluate(model, config, docs, vocab, tokenizer, store, split, device, facts,
              condition='clean', rate=0.0, noise_seed=101):
    model.eval()
    predictions, observations, doc_results = [], [], []
    evidence_tp = evidence_gold = evidence_pred = evidence_pairs = total_pairs = 0
    threshold_tp = threshold_pred = 0
    start = time.perf_counter()
    for doc in docs:
        observation = observation_for(doc, condition, rate, noise_seed, store, split)
        feature, inputs, targets = _batch(doc, tokenizer, vocab, observation, config, device)
        # 评价时不向作者模型提供关系、证据或参考类型标签。
        outputs = model(inputs)
        selected = _official_predictions(doc, feature, outputs['rel_pred'], vocab)
        predictions.extend(selected)
        doc_results.append({'doc_id': doc['doc_id'], 'predictions': selected})
        observations.append(observation)
        evidence = outputs.get('evidence_scores')
        total_pairs += len(feature['hts'])
        if evidence is not None:
            for index, valid in enumerate(feature['evidence_mask']):
                if not valid:
                    continue
                evidence_pairs += 1
                truth = {s for s, value in enumerate(feature['sent_labels'][index]) if value > 0}
                predicted = set(evidence[index].topk(min(2, len(doc['sents']))).indices.tolist())
                threshold_selected = set(torch.where(evidence[index] > config.get('etmier', {}).get('evidence_threshold', .2))[0].tolist())
                evidence_tp += len(truth & predicted)
                evidence_gold += len(truth)
                evidence_pred += len(predicted)
                threshold_tp += len(truth & threshold_selected)
                threshold_pred += len(threshold_selected)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    metrics = evaluate_relations(docs, predictions, facts)
    metrics['evidence'] = {'labeled_pairs': evidence_pairs, 'candidate_pairs': total_pairs,
                           'top_k': 2,
                           'precision': evidence_tp / max(evidence_pred, 1),
                           'recall': evidence_tp / max(evidence_gold, 1),
                           'f1': 2 * evidence_tp / max(evidence_pred + evidence_gold, 1),
                           'scope': '仅非空证据标注实体对；未完备参考'}
    metrics['evidence_upstream_threshold'] = {
        'threshold': config.get('etmier', {}).get('evidence_threshold', .2),
        'labeled_pairs': evidence_pairs, 'precision': threshold_tp / max(threshold_pred, 1),
        'recall': threshold_tp / max(evidence_gold, 1),
        'f1': 2 * threshold_tp / max(threshold_pred + evidence_gold, 1),
        'scope': '作者阈值规则的补充指标；主 evidence 字段共用 top-2 规则',
    }
    metrics['type_input'] = {'consumed_for_relation_prediction': False,
                             'training_auxiliary_target': 'clean_consistent_reference',
                             'interpretation': '类型标签仅用于辅助训练监督；输入类型扰动曲线应保持不变，不代表纠错能力'}
    metrics['diagnostics'] = {'seconds': time.perf_counter() - start,
                              'timing_scope': '逐文档特征整理、前向与指标汇总，不是预热模型延时'}
    return {'metrics': metrics, 'predictions': predictions, 'documents': doc_results,
            'observations': observations, 'condition': condition, 'rate': rate,
            'noise_seed': noise_seed, 'split': split, 'doc_ids': [d['doc_id'] for d in docs],
            'training_seed': config['training']['seed'], 'variant': 'E1',
            'source_sha256': sha256(config['data'][split]),
            'decision_rule': 'upstream adaptive NA threshold with fixed max labels',
            'protocol': 'single_pass_controlled; no inference_stage_fusion'}


def train(config, output=None, device='auto', resume=None):
    from .training import run_training

    config = copy.deepcopy(config)
    settings = config['training']
    if int(settings['epochs']) < 1 or int(settings.get('accumulation_steps', 1)) < 1:
        raise ValueError('epochs 和 accumulation_steps 必须为正整数。')
    if settings.get('max_steps') is not None and int(settings['max_steps']) < 1:
        raise ValueError('max_steps 必须为正整数。')
    seed_everything(settings['seed'], settings.get('threads', 2))
    device = choose_device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if resume is not None:
        run_dir, _ = resolve_resume_path(resume)
        if output is not None and resolve_path(output).resolve() != run_dir.resolve():
            raise ValueError('断点续训必须使用原运行目录；请移除 --output 或指定同一目录。')
    else:
        run_dir = resolve_path(output) if output else resolve_path(config['output']['directory']) / (
            'E1_seed%d_%s' % (settings['seed'], time.strftime('%Y%m%d_%H%M%S')))
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError('E1 运行目录非空，请使用新目录。')
    run_dir.mkdir(parents=True, exist_ok=True)
    all_train = load_split(config['data']['train'])
    vocab = build_relation_vocab(all_train)
    train_docs = all_train[:settings.get('limit_train')] if settings.get('limit_train') else all_train
    dev_docs = load_split(config['data']['dev'], settings.get('limit_dev'))
    if not train_docs or not dev_docs:
        raise ValueError('训练/开发文档不能为空。')
    store = TeacherStore(config)
    if store.metadata is None or store.confusion is None:
        raise ValueError('统一干预/选模需要已准备的 train OOF 类型缓存与来源清单。')
    store.get('train', train_docs[0]['doc_id'])
    config['teacher']['expected_manifest_sha256'] = sha256(store.directory / 'manifest.json')
    tokenizer = create_tokenizer(config)
    model = ETMIERAdapter(config, len(vocab), tokenizer=tokenizer).to(device)
    config.setdefault('etmier', {})['transformer_type'] = model.encoder.config.transformer_type
    backbone, added = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith('official.model.') else added).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': settings['learning_rate']},
        {'params': added, 'lr': settings.get('head_learning_rate', settings['learning_rate'])},
    ], weight_decay=settings.get('weight_decay', .01))
    facts = build_train_facts(all_train)
    metadata = {'config': config, 'relation_vocab': vocab, 'environment': environment_info(),
                'data_sha256': {s: sha256(config['data'][s]) for s in ('train', 'dev', 'test')},
                'teacher_manifest_sha256': config['teacher']['expected_manifest_sha256'],
                'source_files_sha256': source_manifest(), 'upstream_provenance': model.provenance,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'microbatch_documents': 1, 'effective_batch_documents': settings['accumulation_steps'],
                'type_input_consumed_for_prediction': False, 'status': 'running',
                'result_scope': 'smoke_only' if config['model']['encoder_backend'] == 'tiny' else 'controlled_single_pass_baseline'}

    def training_step(index, epoch):
        doc = train_docs[index]
        observation = training_observation(doc, config, epoch, store)
        _, inputs, targets = _batch(doc, tokenizer, vocab, observation, config, device)
        outputs = model(inputs, targets)
        audit = {'doc_id': doc['doc_id'], 'observation': observation,
                 'auxiliary_type_target': 'unchanged clean reference; inconsistent masked'}
        return outputs['loss'], audit

    def evaluate_epoch():
        clean = _evaluate(model, config, dev_docs, vocab, tokenizer, store, 'dev', device, facts)
        noisy = _evaluate(model, config, dev_docs, vocab, tokenizer, store, 'dev', device, facts,
                          'confusion', .2, config['evaluation'].get('selection_noise_seed', 101))
        return clean, noisy

    # E1 仅接入共享训练状态管理，作者模型、损失与选模规则保持不变。
    return run_training(model, optimizer, train_docs, config, run_dir, metadata,
                        training_step, evaluate_epoch, resume=resume,
                        checkpoint_extra={'upstream_provenance': model.provenance})


def load_checkpoint(path, device='auto'):
    path = Path(path).resolve()
    path = path / 'best' if (path / 'best').exists() else path
    metadata = read_json(path / 'checkpoint.json')
    config, vocab = relocate_config(metadata['config']), metadata['relation_vocab']
    for split, expected in metadata['data_sha256'].items():
        if sha256(config['data'][split]) != expected:
            raise ValueError('E1 当前数据与 checkpoint 不一致：' + split)
    config['teacher']['expected_manifest_sha256'] = metadata['teacher_manifest_sha256']
    TeacherStore(config)
    seed_everything(config['training']['seed'], config['training'].get('threads', 2))
    device = choose_device(device)
    model = ETMIERAdapter(config, len(vocab), tokenizer=create_tokenizer(config)).to(device)
    current = {key: value for key, value in model.provenance.items() if key != 'source_dir'}
    saved = {key: value for key, value in metadata['upstream_provenance'].items() if key != 'source_dir'}
    if current != saved:
        raise ValueError('E1 作者源码或兼容适配版本与 checkpoint 不一致。')
    model.load_state_dict(load_file(str(path / 'model.safetensors'), device=str(device)))
    return model.eval(), config, vocab, device


def evaluate(checkpoint, split='test', condition='clean', rate=0.0, noise_seed=101,
             output=None, device='auto', limit=None):
    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs = load_split(config['data'][split], limit)
    facts = build_train_facts(load_split(config['data']['train']))
    result = _evaluate(model, config, docs, vocab, create_tokenizer(config), TeacherStore(config),
                        split, device, facts, condition, rate, noise_seed)
    result['checkpoint'] = str(Path(checkpoint).resolve())
    if output:
        write_json(output, result)
        write_json(Path(output).with_suffix('.predictions.json'), result['predictions'])
    return result


def sweep(checkpoint, split, output, device='auto', limit=None):
    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs = load_split(config['data'][split], limit)
    facts = build_train_facts(load_split(config['data']['train']))
    tokenizer, store = create_tokenizer(config), TeacherStore(config)
    conditions = [('clean', 0., 101), ('local', 0., 101)]
    conditions += [(name, rate, seed) for name in ('missing', 'uniform', 'confusion')
                   for rate in config['evaluation']['rates'] for seed in config['evaluation']['noise_seeds']]
    results = []
    reference_predictions = None
    for name, rate, seed in conditions:
        result = _evaluate(model, config, docs, vocab, tokenizer, store, split, device, facts, name, rate, seed)
        if reference_predictions is None:
            reference_predictions = result['predictions']
        invariant = result['predictions'] == reference_predictions
        write_json(Path(output) / ('%s_%02d_seed%d.json' % (name, round(rate * 100), seed)), result)
        results.append({'condition': name, 'rate': rate, 'noise_seed': seed,
                        'f1': result['metrics']['f1'], 'ign_f1': result['metrics']['ign_f1'],
                        'predictions_equal_clean': invariant})
    summary = {'variant': 'E1', 'rows': results, 'expected_type_input_invariance': True,
               'interpretation': '作者推理未消费外部类型，平坦曲线不能证明纠错效果'}
    write_json(Path(output) / 'summary.json', summary)
    if not all(row['predictions_equal_clean'] for row in results):
        raise AssertionError('E1 外部类型改变后预测不一致，请检查随机性或输入变化。')
    return summary


@torch.no_grad()
def diagnose(checkpoint, split, output, device='auto', limit=10):
    model, config, vocab, device = load_checkpoint(checkpoint, device)
    tokenizer, store = create_tokenizer(config), TeacherStore(config)
    rows = []
    for doc in load_split(config['data'][split], limit):
        obs = observation_for(doc, 'clean', 0., 101, store, split)
        _, inputs, _ = _batch(doc, tokenizer, vocab, obs, config, device)
        original = model(inputs)['logits']
        changed_obs = observation_for(doc, 'uniform', 1., 101, store, split)
        _, changed_inputs, _ = _batch(doc, tokenizer, vocab, changed_obs, config, device)
        difference = float((original - model(changed_inputs)['logits']).abs().max())
        rows.append({'doc_id': doc['doc_id'], 'max_logit_difference': difference})
    result = {'variant': 'E1', 'passed': all(row['max_logit_difference'] <= 1e-6 for row in rows),
              'rows': rows, 'scope': '外部类型未进入关系推理的完整性验证，不是关闭 G 的回退测试'}
    write_json(output, result)
    if not result['passed']:
        raise AssertionError('E1 类型输入不变性检查失败。')
    return result
