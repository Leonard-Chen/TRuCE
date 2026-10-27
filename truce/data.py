#!/usr/bin/env python3
'''Re-DocRED 输入、类型无关构图与长文档分块。'''

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


TYPE_NAMES = ('PER', 'ORG', 'LOC', 'NUM', 'TIME', 'MISC')
TYPE_TO_ID = {name: index for index, name in enumerate(TYPE_NAMES)}


def load_documents(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        documents = json.load(stream)
    if not isinstance(documents, list):
        raise ValueError('数据文件必须是官方文档列表 JSON')
    return documents


def document_fingerprint(document):
    payload = {key: document.get(key) for key in ('title', 'sents', 'vertexSet', 'labels')}
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def normalize_document(raw, doc_id):
    doc = copy.deepcopy(raw)
    doc['doc_id'] = str(doc_id)
    doc['title'] = str(doc.get('title', doc_id))
    doc.setdefault('labels', [])
    sentences = doc['sents']
    entities = doc['vertexSet']
    if not sentences or not entities:
        raise ValueError('文档须包含句子与实体')
    reference_types, consistent, entity_sentences = [], [], []
    for entity in entities:
        if not entity:
            raise ValueError('实体簇不能为空')
        counts = Counter()
        sentence_ids = set()
        for mention in entity:
            if mention['type'] not in TYPE_TO_ID:
                raise ValueError('未知实体类型：{}'.format(mention['type']))
            sid = int(mention['sent_id'])
            begin, end = (int(value) for value in mention['pos'])
            if not (0 <= sid < len(sentences) and 0 <= begin < end <= len(sentences[sid])):
                raise ValueError('实体提及位置无效')
            counts[TYPE_TO_ID[mention['type']]] += 1
            sentence_ids.add(sid)
        # 多数投票；平票按固定类序处理。原始 mention 类型只保留为审计材料。
        reference_types.append(min(counts, key=lambda index: (-counts[index], index)))
        consistent.append(len(counts) == 1)
        entity_sentences.append(sentence_ids)
    size = len(entities)
    adjacency = []
    for target in range(size):
        row = [float(target == source or bool(entity_sentences[target] & entity_sentences[source]))
               for source in range(size)]
        denominator = sum(row)
        adjacency.append([value / denominator for value in row])
    for label in doc['labels']:
        if not (0 <= int(label['h']) < size and 0 <= int(label['t']) < size):
            raise ValueError('关系引用未知实体')
        if int(label['h']) == int(label['t']):
            raise ValueError('不支持实体自关系')
        if any(not 0 <= int(sid) < len(sentences) for sid in label.get('evidence', [])):
            raise ValueError('证据句索引无效')
    doc.update(reference_types=reference_types, class_consistent=consistent,
               adjacency=adjacency,
               pairs=[(h, t) for h in range(size) for t in range(size) if h != t],
               source_fingerprint=document_fingerprint(raw))
    return doc


def build_relation_vocab(train_documents):
    names = {str(label['r']) for document in train_documents for label in document.get('labels', [])}
    names.difference_update({'NA', 'Na', 'no_relation'})
    if not names:
        raise ValueError('训练数据没有正关系类别')
    return {name: index for index, name in enumerate(sorted(names))}


def _word_ids(tokenizer, word):
    ids = tokenizer.encode(str(word), add_special_tokens=False)
    return ids if ids else [tokenizer.unk_token_id]


def encode_document(doc, tokenizer, relation_vocab, max_length=512, overlap=128):
    if sorted(relation_vocab.values()) != list(range(len(relation_vocab))):
        raise ValueError('relation_vocab 必须为从 0 开始的连续索引')
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    if special_count != 2:
        raise ValueError('当前编码器要求单句 BERT [CLS]/[SEP] 格式')
    capacity = max_length - special_count
    if capacity <= 0 or not 0 <= overlap < capacity:
        raise ValueError('需满足 0 <= overlap < max_length - 特殊符号数')
    marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in ('[unused0]', '[unused1]')]
    if len(set(marker_ids)) != 2 or any(value is None or value == tokenizer.unk_token_id for value in marker_ids):
        raise ValueError('本项目需支持 [unused0]/[unused1] 的 BERT tokenizer')
    begins, ends = set(), set()
    for entity in doc['vertexSet']:
        for mention in entity:
            sid = int(mention['sent_id'])
            begin, end = mention['pos']
            begins.add((sid, begin))
            ends.add((sid, end))
    tokens, word_spans, sentence_spans = [], {}, []
    for sid, sentence in enumerate(doc['sents']):
        sentence_start = len(tokens)
        for wid, word in enumerate(sentence):
            if (sid, wid) in ends:
                tokens.append(marker_ids[1])
            if (sid, wid) in begins:
                tokens.append(marker_ids[0])
            begin = len(tokens)
            tokens.extend(_word_ids(tokenizer, word))
            word_spans[(sid, wid)] = (begin, len(tokens))
        if (sid, len(sentence)) in ends:
            tokens.append(marker_ids[1])
        sentence_spans.append((sentence_start, len(tokens)))
    if not tokens:
        raise ValueError('文档没有可编码 token')
    entity_spans = []
    for entity in doc['vertexSet']:
        spans = []
        for mention in entity:
            sid = int(mention['sent_id'])
            begin, end = mention['pos']
            # 按原词分别保存区间，避免嵌套/交叠提及的边界标记混入实体池化。
            spans.extend(word_spans[(sid, wid)] for wid in range(begin, end))
        entity_spans.append(spans)
    chunks = []
    start = 0
    while start < len(tokens):
        stop = min(start + capacity, len(tokens))
        part = tokens[start:stop]
        input_ids = tokenizer.build_inputs_with_special_tokens(part)
        token_map = [-1] + list(range(start, stop)) + [-1]
        if (input_ids[0] != tokenizer.cls_token_id or input_ids[-1] != tokenizer.sep_token_id
                or input_ids[1:-1] != part or len(input_ids) != len(token_map)):
            raise ValueError('tokenizer 的特殊符号映射不一致')
        chunks.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids),
                       'token_map': token_map})
        if stop == len(tokens):
            break
        start += capacity - overlap
    pair_to_index = {pair: index for index, pair in enumerate(doc['pairs'])}
    relation_targets = [[0.0] * len(relation_vocab) for _ in doc['pairs']]
    evidence_targets = [[0.0] * len(doc['sents']) for _ in doc['pairs']]
    evidence_mask = [False] * len(doc['pairs'])
    for label in doc['labels']:
        relation = str(label['r'])
        if relation not in relation_vocab:
            raise ValueError('关系 {} 不在训练集关系表；不得从开发/测试集扩展 schema'.format(relation))
        index = pair_to_index[(int(label['h']), int(label['t']))]
        relation_targets[index][relation_vocab[relation]] = 1.0
        for sid in label.get('evidence', []):
            evidence_targets[index][int(sid)] = 1.0
            evidence_mask[index] = True
    return {
        'doc_id': doc['doc_id'], 'title': doc['title'], 'chunks': chunks,
        'num_tokens': len(tokens), 'num_relations': len(relation_vocab), 'entity_spans': entity_spans,
        'sentence_spans': sentence_spans, 'adjacency': doc['adjacency'], 'pairs': doc['pairs'],
        'relation_targets': relation_targets, 'evidence_targets': evidence_targets,
        'evidence_mask': evidence_mask,
        'targets': {'reference_types': doc['reference_types'],
                    'class_consistent': doc['class_consistent']},
        'source_fingerprint': doc['source_fingerprint'],
    }
