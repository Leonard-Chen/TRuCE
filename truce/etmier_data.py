#!/usr/bin/env python3
'''ET-MIER Re-DocRED 分支的输入适配，保留官方实体标记与候选对顺序'''

import torch

from .data import TYPE_NAMES
from .interventions import make_observation


ETMIER_TYPE_TO_ID = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3,
                     'TIME': 4, 'MISC': 5, 'PER': 6}
CANONICAL_TO_ETMIER = tuple(ETMIER_TYPE_TO_ID[name] for name in TYPE_NAMES)


def encode_etmier_document(doc, tokenizer, relation_vocab, observation=None,
                          max_seq_length=1024, transformer_type='bert'):
    if sorted(relation_vocab.values()) != list(range(len(relation_vocab))):
        raise ValueError('relation_vocab 必须为从 0 开始的连续正关系索引')
    if any(name in relation_vocab for name in ('NA', 'Na', 'no_relation')):
        raise ValueError('输入关系表只包含正关系；ET-MIER 的 NA 由适配器加入索引 0')
    if transformer_type not in ('bert', 'roberta'):
        raise ValueError('ET-MIER 只支持 bert 或 roberta')
    if tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError('ET-MIER 单文档编码要求首尾共两个特殊 token')
    if len(doc['vertexSet']) < 2 or any(not sent for sent in doc['sents']):
        raise ValueError('ET-MIER 文档须包含至少两个实体，且句子不能为空')
    observation = observation if observation is not None else make_observation(doc)
    observed = list(observation['observed_types'])
    if len(observed) != len(doc['vertexSet']) or any(
            not isinstance(value, int) or not -1 <= value < len(TYPE_NAMES)
            for value in observed):
        raise ValueError('观测类型数量须与实体一致，取值须为 -1 或 0..5')
    observed_upstream = [0 if value == -1 else CANONICAL_TO_ETMIER[value]
                         for value in observed]
    references = [CANONICAL_TO_ETMIER[value] for value in doc['reference_types']]
    consistent = doc['class_consistent']
    starts, ends = set(), set()
    for entity in doc['vertexSet']:
        for mention in entity:
            sid, (begin, end) = int(mention['sent_id']), mention['pos']
            starts.add((sid, begin))
            ends.add((sid, end - 1))

    tokens, sent_map, sent_pos = [], [], []
    for sid, sentence in enumerate(doc['sents']):
        start = len(tokens)
        mapping = {}
        for wid, word in enumerate(sentence):
            wordpieces = tokenizer.tokenize(word)
            if (sid, wid) in starts:
                wordpieces = ['*'] + wordpieces
            if (sid, wid) in ends:
                wordpieces = wordpieces + ['*']
            mapping[wid] = len(tokens)
            tokens.extend(wordpieces)
        mapping[len(sentence)] = len(tokens)
        sent_map.append(mapping)
        sent_pos.append((start, len(tokens)))
    input_ids = tokenizer.build_inputs_with_special_tokens(
        tokenizer.convert_tokens_to_ids(tokens))
    # 官方双片段方案在上限附近会漏掉中间 token；显式拒绝，避免悄悄截断实体与证据
    coverage_limit = 1022 if transformer_type == 'bert' else 1021
    limit = min(int(max_seq_length), coverage_limit)
    if len(input_ids) > limit:
        raise ValueError('文档 {} 含 {} 个编码 token，超过 ET-MIER 无缺口上限 {}；'
                         '请显式选择长文档处理方案，不得静默截断'.format(
                             doc['doc_id'], len(input_ids), limit))
    entity_pos = [[(sent_map[int(m['sent_id'])][int(m['pos'][0])],
                    sent_map[int(m['sent_id'])][int(m['pos'][1])])
                   for m in entity] for entity in doc['vertexSet']]

    triples = {}
    for label in doc.get('labels', []):
        relation = str(label['r'])
        if relation not in relation_vocab:
            raise ValueError('关系 {} 不在训练集关系表'.format(relation))
        pair = (int(label['h']), int(label['t']))
        triples.setdefault(pair, []).append(label)
    # 上游按标注首次出现顺序排列正样本，再追加所有剩余有向实体对。
    pairs = list(triples)
    pairs.extend((h, t) for h in range(len(entity_pos)) for t in range(len(entity_pos))
                 if h != t and (h, t) not in triples)
    labels, sent_labels = [], []
    for pair in pairs:
        relation_target = [0] * (len(relation_vocab) + 1)
        evidence_target = [0] * len(sent_pos)
        for label in triples.get(pair, []):
            relation_target[relation_vocab[str(label['r'])] + 1] = 1
            for sid in label.get('evidence', []):
                evidence_target[int(sid)] += 1
        if pair not in triples:
            relation_target[0] = 1
        labels.append(relation_target)
        sent_labels.append(evidence_target)
    return {
        'doc_id': doc['doc_id'], 'title': doc['title'], 'input_ids': input_ids,
        'entity_pos': entity_pos, 'labels': labels, 'hts': [list(pair) for pair in pairs],
        'pairs': pairs, 'sent_pos': sent_pos, 'sent_labels': sent_labels,
        'epair_types': [[observed_upstream[h], observed_upstream[t]] for h, t in pairs],
        'observed_types': observed, 'observed_entity_types': observed_upstream,
        # 观测类型只用于审计；上游 Re-DocRED 分支的类型损失使用独立参考标签。
        'reference_epair_types': [[references[h], references[t]] for h, t in pairs],
        'reference_epair_mask': [[bool(consistent[h]), bool(consistent[t])] for h, t in pairs],
        'evidence_mask': [any(row) for row in sent_labels],
        'targets': {'reference_types': list(doc['reference_types']),
                    'class_consistent': list(consistent)},
        'source_fingerprint': doc['source_fingerprint'],
        'num_tokens': len(tokens), 'num_relations': len(relation_vocab),
        'max_supported_input_length': limit,
    }


def collate_etmier(features, device=None, pad_token_id=0):
    if not features:
        raise ValueError('ET-MIER 批次不能为空')
    max_tokens = max(len(feature['input_ids']) for feature in features)
    max_sentences = max(len(feature['sent_pos']) for feature in features)
    relation_dims = {len(feature['labels'][0]) for feature in features}
    if len(relation_dims) != 1:
        raise ValueError('同一批次必须使用同一关系表')
    ids, masks, sentence_labels = [], [], []
    for feature in features:
        padding = max_tokens - len(feature['input_ids'])
        ids.append(feature['input_ids'] + [int(pad_token_id)] * padding)
        masks.append([1.0] * len(feature['input_ids']) + [0.0] * padding)
        sentence_labels.extend(row + [0] * (max_sentences - len(row))
                               for row in feature['sent_labels'])
    output = {
        'input_ids': torch.tensor(ids, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(masks, dtype=torch.float, device=device),
        'labels': torch.tensor([row for feature in features for row in feature['labels']],
                               dtype=torch.long, device=device),
        'entity_pos': [feature['entity_pos'] for feature in features],
        'hts': [feature['hts'] for feature in features],
        'sent_pos': [feature['sent_pos'] for feature in features],
        'sent_labels': torch.tensor(sentence_labels, dtype=torch.long, device=device),
        'attns': None,
    }
    for field in ('epair_types', 'reference_epair_types', 'reference_epair_mask', 'evidence_mask'):
        dtype = torch.bool if field.endswith('mask') else torch.long
        output[field] = torch.tensor([row for feature in features for row in feature[field]],
                                     dtype=dtype, device=device)
    return output
