#!/usr/bin/env python3
'''只修改观测类型；参考标签与扰动真值仅用于监督和审计。'''

import hashlib
import math

from .data import TYPE_NAMES, document_fingerprint


def stable_uniform(*parts):
    value = '\x1f'.join(str(part) for part in parts).encode('utf-8')
    return (int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') >> 11) / float(2 ** 53)


def build_confusion_matrix(reference_types, predictions, consistent=None):
    if len(reference_types) != len(predictions):
        raise ValueError('参考类型与预测数量不同')

    if consistent is None:
        consistent = [True] * len(reference_types)

    if len(consistent) != len(reference_types):
        raise ValueError('一致性标志数量不同')

    matrix = [[float(row != col) for col in range(6)] for row in range(6)]
    for reference, prediction, valid in zip(reference_types, predictions, consistent):
        prediction = _argmax(prediction) if isinstance(prediction, (list, tuple)) else int(prediction)
        reference = int(reference)
        if not 0 <= reference < 6 or not 0 <= prediction < 6:
            raise ValueError('类型索引须为 0..5')
        if valid and reference != prediction:
            matrix[reference][prediction] += 1.0

    return [[value / sum(row) for value in row] for row in matrix]


def _argmax(values):
    if len(values) != 6 or any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError('类型概率须为 6 个非负有限数')
    if sum(values) <= 0:
        raise ValueError('类型概率之和须大于零')
    return max(range(6), key=lambda index: values[index])


def _sample_wrong(reference, uniform, confusion):
    weights = [float(index != reference) for index in range(6)]
    if confusion is not None:

        if len(confusion) != 6 or any(len(row) != 6 for row in confusion):
            raise ValueError('混淆矩阵须为 6×6')

        weights = [0.0 if index == reference else float(confusion[reference][index]) for index in range(6)]
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError('混淆矩阵须为有限非负值')

    total = sum(weights)
    if total <= 0:
        raise ValueError('混淆矩阵的非对角元素之和须大于零')

    cutoff = uniform * total
    accumulator = 0.0
    for index, weight in enumerate(weights):
        accumulator += weight
        if weight > 0 and cutoff < accumulator:
            return index

    return max(index for index, weight in enumerate(weights) if weight > 0)


def make_observation(doc, condition='clean', rate=0.0, seed=101, confusion=None, local_probs=None):
    aliases = {'C0': 'clean', 'C1': 'missing', 'C2': 'uniform', 'C3': 'confusion', 'C4': 'local'}
    condition = aliases.get(condition, condition)
    if condition not in {'clean', 'missing', 'uniform', 'confusion', 'local'}:
        raise ValueError('未知干预条件：{}'.format(condition))

    if not 0 <= rate <= 1:
        raise ValueError('噪声比例须介于 0 和 1')

    if condition == 'confusion' and confusion is None:
        raise ValueError('混淆错标需要仅由 train OOF 估计的混淆矩阵')

    references = doc['reference_types']
    if condition == 'local' and (local_probs is None or len(local_probs) != len(references)):
        raise ValueError('C4 需要当前文档每个实体的固定 q_local 概率')

    observed, manifest_entities = [], []
    for entity_id, reference in enumerate(references):
        # 选中随机数与替换随机数均不依赖 rho，保证强度递增时 mask 嵌套
        selection = stable_uniform(seed, doc['doc_id'], entity_id, 'selection')
        replacement_u = stable_uniform(seed, doc['doc_id'], entity_id, condition, 'replacement')
        chosen = selection < rate
        value = int(reference)
        replacement = None
        if condition == 'local':
            value = _argmax(local_probs[entity_id])
        elif condition == 'missing' and chosen:
            value = -1
        elif condition in {'uniform', 'confusion'}:
            replacement = _sample_wrong(reference, replacement_u, confusion if condition == 'confusion' else None)
            if chosen:
                value = replacement

        observed.append(value)
        manifest_entities.append({'entity_id': entity_id, 'reference_type': reference,
                                  'reference_consistent': doc['class_consistent'][entity_id],
                                  'observed_type': value, 'was_corrupted': value != reference,
                                  'is_missing': value == -1, 'selection_u': selection,
                                  'preselected_wrong_type': replacement})

    actual_rate = sum(value != reference for value, reference in zip(observed, references)) / max(1, len(references))
    return {'observed_types': observed, 'missing': [value == -1 for value in observed],
            'manifest': {'doc_id': doc['doc_id'], 'condition': condition, 'rate': rate,
                         'seed': seed, 'type_order': list(TYPE_NAMES), 'actual_rate': actual_rate,
                         'source_fingerprint': doc.get('source_fingerprint', document_fingerprint(doc)),
                         'entities': manifest_entities}}


def make_training_observation(doc, epoch, seed=17, confusion=None, corruption_rate=0.2):
    draw = stable_uniform(seed, epoch, doc['doc_id'], 'training_mixture')
    condition = 'clean' if draw < 0.5 else 'missing' if draw < 0.75 else 'uniform' if draw < 0.875 else 'confusion'
    if condition == 'confusion' and confusion is None:
        raise ValueError('混合训练需要 train OOF 混淆矩阵；不要静默改为均匀错标')

    result = make_observation(doc, condition, 0.0 if condition == 'clean' else corruption_rate,
                              seed='{}:epoch:{}'.format(seed, epoch), confusion=confusion)
    result['manifest']['training_epoch'] = epoch
    result['manifest']['training_seed'] = seed
    return result


def mask_evidence_sentences(raw_document, sentence_ids, keep_only=False, mask_token='[MASK]'):
    '''保持实体和 token 位置的诊断性屏蔽；调用方必须清空全文特征缓存'''
    import copy
    document = copy.deepcopy(raw_document)
    selected = set(int(index) for index in sentence_ids)
    if any(index < 0 or index >= len(document['sents']) for index in selected):
        raise ValueError('证据句索引无效')

    mentions = set()
    for entity in document['vertexSet']:
        for mention in entity:
            mentions.update((mention['sent_id'], wid) for wid in range(*mention['pos']))

    for sid, sentence in enumerate(document['sents']):
        should_mask = sid not in selected if keep_only else sid in selected
        if should_mask:
            document['sents'][sid] = [word if (sid, wid) in mentions else mask_token
                                      for wid, word in enumerate(sentence)]

    return document
