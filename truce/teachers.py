#!/usr/bin/env python3
'''冻结文本特征、文档分折类型器与可追踪的类型预测缓存。'''

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import TYPE_NAMES, encode_document, normalize_document
from .runtime import choose_device, load_split, read_json, seed_everything, sha256, tensorize, write_json
from .tokenization import create_tokenizer


def split_document_folds(doc_ids, seed=17, folds=2):
    if folds != 2:
        raise ValueError('实验方案固定为文档级两折 OOF，暂不支持自定义折数')

    ids = sorted(str(value) for value in doc_ids)
    if len(ids) < 4 or len(ids) != len(set(ids)):
        raise ValueError('OOF 至少需要 4 个身份唯一的训练文档')

    rng = np.random.RandomState(seed)
    order = [ids[index] for index in rng.permutation(len(ids))]
    return [order[index::folds] for index in range(folds)]


def local_window_document(document, entity_index, radius=16):
    offsets, cursor = [], 0
    for sentence in document['sents']:
        offsets.append(cursor)
        cursor += len(sentence)

    words = [word for sentence in document['sents'] for word in sentence]
    mention = min(document['vertexSet'][entity_index],
                  key=lambda item: (int(item['sent_id']), int(item['pos'][0])))
    begin = offsets[int(mention['sent_id'])] + int(mention['pos'][0])
    end = offsets[int(mention['sent_id'])] + int(mention['pos'][1])
    left, right = max(0, begin - radius), min(len(words), end + radius)
    # 先裁原词，再独立编码；禁止从已看过全文的 hidden states 裁窗口
    raw = {'title': '{}::local::{}'.format(document['doc_id'], entity_index),
           'sents': [words[left:right]], 'labels': [], 'vertexSet': [[{
               'name': mention.get('name', ' '.join(words[begin:end])),
               'sent_id': 0, 'pos': [begin - left, end - left], 'type': 'PER'}]]}

    return normalize_document(raw, raw['title'])


class TypeHead(nn.Module):
    '''
    类型头
    '''
    def __init__(self, feature_dim, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_size), nn.Tanh(),
                                     nn.Linear(hidden_size, len(TYPE_NAMES)))
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('scale', torch.ones(feature_dim))
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

    def forward(self, values):
        return self.network((values - self.mean) / self.scale)


def _stack_supervision(features, labels, consistent, ids):
    values, targets = [], []
    for doc_id in ids:
        mask = torch.as_tensor(consistent[doc_id], dtype=torch.bool)
        values.append(torch.as_tensor(features[doc_id], dtype=torch.float32)[mask])
        targets.append(torch.as_tensor(labels[doc_id], dtype=torch.long)[mask])

    x, y = torch.cat(values), torch.cat(targets)
    if not len(y):
        raise ValueError('当前训练/校准分折没有类型一致的实体，不能拟合类型器')
    return x, y


def _fit_head(features, labels, consistent, ids, settings, seed):
    x, y = _stack_supervision(features, labels, consistent, ids)
    torch.manual_seed(seed)

    head = TypeHead(x.shape[1], int(settings.get('hidden', 64)))
    head.mean.copy_(x.mean(dim=0))
    head.scale.copy_(x.std(dim=0, unbiased=False).clamp_min(1e-5))
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(settings.get('lr', 0.01)),
                                  weight_decay=float(settings.get('weight_decay', 0.01)))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(settings.get('batch_size', 512))
    epochs = int(settings.get('epochs', 30))
    if epochs < 1 or batch_size < 1:
        raise ValueError('类型头 epochs 和 batch_size 必须为正整数')

    head.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(y), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[indices]), y[indices])
            loss.backward()
            optimizer.step()

    return head.eval()


def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if not len(labels):
        raise ValueError('温度校准集不能为空')
    with torch.no_grad():
        candidates = torch.logspace(np.log10(0.25), np.log10(8.0), 41)
        losses = torch.stack([F.cross_entropy(logits / value, labels) for value in candidates])
    return float(candidates[int(losses.argmin())])


def fit_oof_type_heads(features, labels, consistent, settings):
    '''这里只接收 train 特征；开发/测试标签没有进入拟合函数的接口。'''
    ids = sorted(features)
    if set(ids) != set(labels) or set(ids) != set(consistent):
        raise ValueError('特征、标签与一致性表的文档身份不一致')

    seed = int(settings.get('seed', 17))
    folds = split_document_folds(ids, seed, int(settings.get('folds', 2)))
    probabilities, raw_logits, records, fitted_heads = {}, {}, [], []
    fraction = float(settings.get('inner_calibration_fraction', 0.2))
    if not 0 < fraction < 1:
        raise ValueError('内部校准比例必须处于 (0, 1)')

    for fold_id, heldout in enumerate(folds):
        outer_train = sorted(set(ids) - set(heldout))
        rng = np.random.RandomState(seed + 100 + fold_id)
        ordered = [outer_train[index] for index in rng.permutation(len(outer_train))]
        count = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
        calibration, fit_ids = ordered[:count], ordered[count:]
        head = _fit_head(features, labels, consistent, fit_ids, settings, seed + fold_id)
        cal_x, cal_y = _stack_supervision(features, labels, consistent, calibration)

        with torch.no_grad():
            temperature = fit_temperature(head(cal_x), cal_y)
            for doc_id in heldout:
                logits = head(torch.as_tensor(features[doc_id], dtype=torch.float32)).detach()
                raw_logits[doc_id] = logits
                probabilities[doc_id] = (logits / temperature).softmax(dim=-1)

        records.append({'fold': fold_id, 'fit_doc_ids': fit_ids,
                        'calibration_doc_ids': calibration, 'heldout_doc_ids': heldout,
                        'temperature': temperature})
        fitted_heads.append(head)

    # 最终头的温度只拟合保存的 OOF logits，不使用全量训练头的 in-sample 预测
    oof_x, oof_y = _stack_supervision(raw_logits, labels, consistent, ids)
    final_temperature = fit_temperature(oof_x, oof_y)
    final_head = _fit_head(features, labels, consistent, ids, settings, seed + 1000)
    return {'probabilities': probabilities, 'raw_logits': raw_logits, 'folds': records,
            'fold_heads': fitted_heads, 'final_head': final_head,
            'final_temperature': final_temperature}


def build_confusion(probabilities, labels, consistent):
    counts = np.zeros((len(TYPE_NAMES), len(TYPE_NAMES)), dtype=np.int64)
    for doc_id, probabilities_for_doc in probabilities.items():
        predictions = torch.as_tensor(probabilities_for_doc).argmax(dim=-1).tolist()
        for reference, predicted, valid in zip(labels[doc_id], predictions, consistent[doc_id]):
            if valid:
                counts[int(reference), int(predicted)] += 1

    smoothed = counts.astype(np.float64) + 1.0
    np.fill_diagonal(smoothed, 0.0)
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return {'type_names': list(TYPE_NAMES), 'counts': counts.tolist(),
            'matrix': smoothed.tolist(), 'source': 'train_q_local_oof',
            'excluded_inconsistent_clusters': True, 'smoothing': 'off_diagonal_add_one'}


class FrozenTextFeatures:
    '''
    冻结后的文本特征
    '''
    def __init__(self, config, device='auto', smoke=False):
        from .model import DocREModel
        self.config = copy.deepcopy(config)
        backend = config.get('model', {}).get('encoder_backend', 'hf')
        if backend == 'tiny' and not smoke:
            raise ValueError('随机 tiny 编码器只允许 --smoke，不得生成正式 teacher')

        self.device = choose_device(str(device))
        seed_everything(int(config.get('teacher', {}).get('seed', 17)))
        self.tokenizer = create_tokenizer(config)
        self.model = DocREModel(config, num_relations=1).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.smoke = smoke

    def _encode(self, document):
        feature_doc = copy.deepcopy(document)
        feature_doc['labels'] = []
        encoded = encode_document(feature_doc, self.tokenizer, {},
                                  self.config['data'].get('max_length', 512),
                                  self.config['data'].get('overlap', 128))
        observation = {'observed_types': [-1] * len(document['vertexSet'])}
        inputs, _ = tensorize(encoded, observation, device=self.device)

        with torch.inference_mode():
            tokens = self.model.encode_tokens(inputs, project=False)
            values = []
            for spans in encoded['entity_spans']:
                positions = sorted({index for begin, end in spans for index in range(begin, end)})
                values.append(tokens[positions].mean(dim=0))

            return torch.stack(values).float().cpu()

    def document(self, document, include_local=True):
        result = {'q_doc': self._encode(document)}
        if include_local:
            radius = int(self.config.get('teacher', {}).get('local_radius', 16))
            result['q_local'] = torch.cat([
                self._encode(local_window_document(document, entity_id, radius))
                for entity_id in range(len(document['vertexSet']))], dim=0)
        return result

    def provenance(self):
        encoder_config = getattr(self.model.encoder, 'config', None)
        revision = getattr(encoder_config, '_commit_hash', None)
        tokenizer = self.tokenizer
        vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {
            'class': type(tokenizer).__name__, 'vocab_size': tokenizer.vocab_size}
        digest = hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode('utf-8')).hexdigest()

        return {'encoder_name': self.config['model'].get('encoder_name', 'bert-base-uncased'),
                'encoder_revision_requested': self.config['model'].get('encoder_revision', 'main'),
                'encoder_revision_resolved': revision,
                'encoder_backend': self.config['model'].get('encoder_backend', 'hf'),
                'tokenizer_class': type(tokenizer).__name__, 'tokenizer_vocab_sha256': digest,
                'feature_type': 'raw_frozen_encoder_hidden_mean_all_mention_tokens',
                'smoke_only': self.smoke,
                'max_length': self.config['data'].get('max_length', 512),
                'overlap': self.config['data'].get('overlap', 128),
                'local_radius_raw_words': self.config.get('teacher', {}).get('local_radius', 16)}


def _save_feature_cache(path, values):
    ids = sorted(values)
    offsets = [0]
    for doc_id in ids:
        offsets.append(offsets[-1] + len(values[doc_id]))

    np.savez_compressed(path, doc_ids=np.asarray(ids), offsets=np.asarray(offsets, dtype=np.int64),
                        features=torch.cat([values[doc_id] for doc_id in ids]).numpy())


def _save_head(path, head):
    from safetensors.torch import save_file
    save_file({key: value.contiguous() for key, value in head.state_dict().items()}, str(path))


def prepare_teacher_artifacts(config, output_dir, smoke=False, device='auto'):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'manifest.json').exists():
        raise FileExistsError('目标已有 teacher manifest，请使用新的目录以保留实验来源：{}'.format(output_dir))

    documents = {split: load_split(config['data'][split]) for split in ('train', 'dev', 'test')}
    source_hashes = {split: sha256(config['data'][split]) for split in documents}
    extractor = FrozenTextFeatures(config, device=device, smoke=smoke)
    provenance = extractor.provenance()
    features = {view: {} for view in ('q_doc', 'q_local')}
    file_manifest = {}

    for split, split_docs in documents.items():

        for view in features:
            features[view][split] = {}

        for index, document in enumerate(split_docs):
            values = extractor.document(document)
            for view in features:
                features[view][split][document['doc_id']] = values[view]
            if (index + 1) % 25 == 0 or index + 1 == len(split_docs):
                print('冻结类型特征 {}: {}/{}'.format(split, index + 1, len(split_docs)), flush=True)

        for view in features:
            path = output_dir / '{}_{}_features.npz'.format(split, view)
            _save_feature_cache(path, features[view][split])
            file_manifest[path.name] = sha256(path)

    del extractor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_docs = documents['train']
    labels = {doc['doc_id']: doc['reference_types'] for doc in train_docs}
    consistent = {doc['doc_id']: doc['class_consistent'] for doc in train_docs}
    fitted, head_metadata = {}, {}
    for view in features:
        fitted[view] = fit_oof_type_heads(features[view]['train'], labels, consistent, config.get('teacher', {}))
        result = fitted[view]

        for fold_id, head in enumerate(result['fold_heads']):
            path = output_dir / '{}_fold_{}.safetensors'.format(view, fold_id)
            _save_head(path, head)
            file_manifest[path.name] = sha256(path)

        path = output_dir / '{}_final.safetensors'.format(view)
        _save_head(path, result['final_head'])
        file_manifest[path.name] = sha256(path)
        head_metadata[view] = {'feature_dim': result['final_head'].feature_dim,
                               'hidden_size': result['final_head'].hidden_size,
                               'temperature': result['final_temperature'],
                               'temperature_source': 'train_oof_raw_logits',
                               'weights': path.name, 'folds': result['folds']}
        logits_path = output_dir / '{}_train_oof_logits.npz'.format(view)
        _save_feature_cache(logits_path, result['raw_logits'])
        file_manifest[logits_path.name] = sha256(logits_path)

    for split, split_docs in documents.items():
        predictions = {}
        with torch.no_grad():

            for document in split_docs:
                doc_id = document['doc_id']
                record = {'source_fingerprint': document['source_fingerprint']}

                for view in features:
                    result = fitted[view]

                    if split == 'train':
                        probs = result['probabilities'][doc_id]
                    else:
                        logits = result['final_head'](features[view][split][doc_id])
                        probs = (logits / result['final_temperature']).softmax(dim=-1)

                    record[view] = probs.tolist()

                predictions[doc_id] = record

        path = output_dir / (split + '.json')
        write_json(path, {'schema_version': 1, 'split': split,
                          'source_sha256': source_hashes[split], 'type_names': list(TYPE_NAMES),
                          'inference_mode': 'oof' if split == 'train' else 'final_head',
                          'smoke_only': smoke, 'docs': predictions})
        file_manifest[path.name] = sha256(path)

    confusion = build_confusion(fitted['q_local']['probabilities'], labels, consistent)
    confusion['source_sha256'] = source_hashes['train']
    write_json(output_dir / 'confusion.json', confusion)
    file_manifest['confusion.json'] = sha256(output_dir / 'confusion.json')
    manifest = {'schema_version': 1, 'status': 'SMOKE_ONLY' if smoke else 'PREPARED',
                'provenance': provenance, 'sources': source_hashes,
                'teacher_settings': copy.deepcopy(config.get('teacher', {})),
                'heads': head_metadata, 'files': file_manifest,
                'training_prediction_mode': 'document_level_oof',
                'dev_test_labels_used_for_fit': False,
                'documents': {split: {doc['doc_id']: {
                    'source_fingerprint': doc['source_fingerprint'],
                    'entity_count': len(doc['vertexSet'])} for doc in values}
                    for split, values in documents.items()}}
    write_json(output_dir / 'manifest.json', manifest)
    return manifest


class TeacherPredictor:
    '''
    用于新文档/C5的重新编码；不会复用原文特征缓存
    '''
    def __init__(self, config, output_dir, device='auto'):
        from safetensors.torch import load_file
        self.output_dir = Path(output_dir)
        self.manifest = read_json(self.output_dir / 'manifest.json')
        expected = self.manifest['provenance']
        smoke = self.manifest['status'] == 'SMOKE_ONLY'

        self.extractor = FrozenTextFeatures(config, device=device, smoke=smoke)
        actual = self.extractor.provenance()

        for key in ('encoder_name', 'encoder_backend', 'tokenizer_vocab_sha256', 'max_length', 'overlap',
                    'local_radius_raw_words', 'encoder_revision_resolved'):
            if actual.get(key) != expected.get(key):
                raise ValueError('teacher 来源不匹配：{}'.format(key))

        self.heads = {}
        for view, metadata in self.manifest['heads'].items():
            path = self.output_dir / metadata['weights']
            if sha256(path) != self.manifest['files'][path.name]:
                raise ValueError('teacher 权重文件 SHA 校验失败')
            head = TypeHead(metadata['feature_dim'], metadata['hidden_size'])
            head.load_state_dict(load_file(str(path)))
            self.heads[view] = head.eval()

    def predict_document(self, document, include_local=False):
        features = self.extractor.document(document, include_local=include_local)
        result = {}
        with torch.no_grad():
            for view, values in features.items():
                temperature = self.manifest['heads'][view]['temperature']
                result[view] = (self.heads[view](values) / temperature).softmax(dim=-1).tolist()
        return result
