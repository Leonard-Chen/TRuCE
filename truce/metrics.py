#!/usr/bin/env python3
'''关系、类型纠正与按文档配对的指标评估，所有比率使用 0~1'''

import random


def build_train_facts(train_documents):
    facts = set()
    for document in train_documents:
        entities = document['vertexSet']
        for label in document.get('labels', []):
            for head in entities[label['h']]:
                for tail in entities[label['t']]:
                    facts.add((head['name'], tail['name'], str(label['r'])))
    return facts


def predictions_from_scores(document, scores, relation_vocab, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('关系概率阈值须介于 0 和 1')

    index_to_relation = {index: relation for relation, index in relation_vocab.items()}
    pairs = document['pairs']
    if len(scores) != len(pairs):
        raise ValueError('预测与实体对数量不同')

    predictions = []
    for (head, tail), row in zip(pairs, scores):
        if len(row) != len(relation_vocab):
            raise ValueError('预测关系维度与训练 schema 不一致')

        for index, score in enumerate(row):
            if float(score) >= threshold:
                predictions.append({'title': document['title'], 'h_idx': int(head),
                                    't_idx': int(tail), 'r': index_to_relation[index]})

    return predictions


def _prediction_key(prediction):
    return (str(prediction['title']), int(prediction['h_idx']), int(prediction['t_idx']), str(prediction['r']))


def _counts_by_document(documents, predictions, train_facts=None):
    train_facts = train_facts or set()
    documents_by_title = {str(document['title']): document for document in documents}
    if len(documents_by_title) != len(documents):
        raise ValueError('文档 title 必须唯一，防止官方格式键碰撞')

    gold = {(str(document['title']), int(label['h']), int(label['t']), str(label['r']))
            for document in documents for label in document.get('labels', [])}
    unique_predictions = {_prediction_key(prediction) for prediction in predictions}
    counts = {title: [0, 0, 0, 0] for title in documents_by_title}

    for title, _, _, _ in gold:
        counts[title][0] += 1
    for key in unique_predictions:
        title, head, tail, relation = key
        counts.setdefault(title, [0, 0, 0, 0])[1] += 1
        if key in gold:
            counts[title][2] += 1
            entities = documents_by_title[title]['vertexSet']
            seen = any((h['name'], t['name'], relation) in train_facts
                       for h in entities[head] for t in entities[tail])
            counts[title][3] += int(seen)

    return counts


def _metrics_from_counts(counts):
    gold, predicted, correct, seen_correct = counts
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # 与官方主评估保持一致：只在精确率中扣除训练事实命中的正确预测，召回率不变
    ign_precision = (correct - seen_correct) / (predicted - seen_correct + 1e-5)
    ign_f1 = 2 * ign_precision * recall / (ign_precision + recall) if ign_precision + recall else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'ign_f1': ign_f1,
            'ign_precision': ign_precision, 'gold': gold, 'predicted': predicted,
            'correct': correct, 'correct_in_train': seen_correct,
            'ign_scope': 'annotated_training_entity_name_facts'}


def evaluate_relations(documents, predictions, train_facts=None):
    counts = _counts_by_document(documents, predictions, train_facts)
    return _metrics_from_counts([sum(row[index] for row in counts.values()) for index in range(4)])


def _quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_bootstrap(documents, predictions_a, predictions_b, train_facts=None, iterations=1000, seed=17):
    if not documents or iterations < 1:
        raise ValueError('配对 bootstrap 需要非空文档且 iterations >= 1')

    counts_a = _counts_by_document(documents, predictions_a, train_facts)
    counts_b = _counts_by_document(documents, predictions_b, train_facts)
    titles = [str(document['title']) for document in documents]
    if set(counts_a) != set(titles) or set(counts_b) != set(titles):
        raise ValueError('bootstrap 预测不得含数据集以外的文档')

    rng = random.Random(seed)
    differences = {'f1': [], 'ign_f1': []}
    for _ in range(iterations):
        sampled = rng.choices(titles, k=len(titles))
        a = _metrics_from_counts([sum(counts_a[title][index] for title in sampled) for index in range(4)])
        b = _metrics_from_counts([sum(counts_b[title][index] for title in sampled) for index in range(4)])

        for name in differences:
            differences[name].append(b[name] - a[name])

    full_a = evaluate_relations(documents, predictions_a, train_facts)
    full_b = evaluate_relations(documents, predictions_b, train_facts)

    return {'direction': 'B minus A', 'unit': 'document', 'iterations': iterations, 'seed': seed,
            'metrics': {name: {'difference': full_b[name] - full_a[name],
                                'ci95': [_quantile(values, 0.025), _quantile(values, 0.975)]}
                        for name, values in differences.items()}}


def type_correction_metrics(reference, observed, corrected, consistent=None):
    size = len(reference)
    if not (len(observed) == len(corrected) == size):
        raise ValueError('类型评估输入长度不同')

    if consistent is None:
        consistent = [True] * size

    if len(consistent) != size:
        raise ValueError('类型一致性标志长度不同')

    predictions = []
    for prediction in corrected:
        if hasattr(prediction, '__len__'):
            if len(prediction) != 6:
                raise ValueError('类型概率维度须为 6')
            # 零向量表示未提供类型，不能由 argmax 的平票规则伪造 PER 恢复
            prediction = max(range(6), key=lambda index: float(prediction[index])) if sum(float(value) for value in prediction) > 0 else -1
        predictions.append(int(prediction))

    counts = {name: 0 for name in ('eligible', 'excluded', 'observed_correct', 'corrected_correct',
                                   'wrong', 'recovered', 'original_correct', 'harmed', 'missing',
                                   'missing_recovered', 'changed', 'changed_to_correct', 'nonmissing')}
    for ref, obs, pred, valid in zip(reference, observed, predictions, consistent):
        if not (0 <= int(ref) < 6 and -1 <= int(obs) < 6 and -1 <= pred < 6):
            raise ValueError('类型索引无效')

        if not valid:
            counts['excluded'] += 1
            continue

        counts['eligible'] += 1
        counts['observed_correct'] += int(obs == ref)
        counts['corrected_correct'] += int(pred == ref)
        if obs == -1:
            counts['missing'] += 1
            counts['missing_recovered'] += int(pred == ref)
            continue
        counts['nonmissing'] += 1
        counts['wrong'] += int(obs != ref)
        counts['recovered'] += int(obs != ref and pred == ref)
        counts['original_correct'] += int(obs == ref)
        counts['harmed'] += int(obs == ref and pred != ref)
        counts['changed'] += int(obs != pred)
        counts['changed_to_correct'] += int(obs != pred and obs != ref and pred == ref)

    rate = lambda numerator, denominator: counts[numerator] / counts[denominator] if counts[denominator] else None

    before, after = rate('observed_correct', 'eligible'), rate('corrected_correct', 'eligible')

    return {'wrong_recovery_rate': rate('recovered', 'wrong'),
            'change_precision': rate('changed_to_correct', 'changed'),
            'change_coverage': rate('changed', 'nonmissing'),
            'correct_to_wrong_rate': rate('harmed', 'original_correct'),
            'missing_recovery_rate': rate('missing_recovered', 'missing'),
            'observed_accuracy': before, 'corrected_accuracy': after,
            'net_accuracy_change': None if before is None else after - before,
            'counts': counts}


def evidence_metrics(targets, probabilities, evidence_mask, threshold=0.5):
    if not (len(targets) == len(probabilities) == len(evidence_mask)):
        raise ValueError('证据预测与标签数量不同')

    true_positive = predicted = gold = eligible = 0
    for truth, scores, valid in zip(targets, probabilities, evidence_mask):
        if len(truth) != len(scores):
            raise ValueError('证据句数量不同')
        if not valid:
            continue
        eligible += 1
        for actual, score in zip(truth, scores):
            selected = float(score) >= threshold
            true_positive += int(bool(actual) and selected)
            predicted += int(selected)
            gold += int(bool(actual))

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0

    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'eligible_pairs': eligible, 'total_pairs': len(targets),
            'coverage': eligible / len(targets) if targets else 0.0,
            'scope': 'available_evidence_annotation_subset'}
