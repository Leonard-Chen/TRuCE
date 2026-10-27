#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch

from .data import encode_document, normalize_document
from .engine import (TEACHER_VARIANTS, TeacherStore, evaluate_model, load_checkpoint,
                     observation_for, prepare, checkpoint_variant)
from .interventions import mask_evidence_sentences
from .metrics import build_train_facts
from .runtime import load_split, sha256, tensorize, write_json
from .tokenization import create_tokenizer


def sweep(checkpoint, split, output, device='auto', limit=None):
    if checkpoint_variant(checkpoint) == 'E1':
        from .etmier_engine import sweep as sweep_etmier
        return sweep_etmier(checkpoint, split, output, device, limit)

    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs, encoded, _ = prepare(config, split, vocab, limit)
    store = TeacherStore(config, model.variant in TEACHER_VARIANTS)
    facts = build_train_facts(load_split(config['data']['train']))
    directory = Path(output)
    conditions = [('clean', 0.0, 101), ('local', 0.0, 101)]
    conditions += [(condition, rate, seed) for condition in ('missing', 'uniform', 'confusion')
                   for rate in config['evaluation']['rates'] for seed in config['evaluation']['noise_seeds']]
    rows = []
    for condition, rate, seed in conditions:
        result = evaluate_model(model, docs, encoded, vocab, config, store, split, device,
                                condition, rate, seed, facts)
        result.update({'split': split, 'source_sha256': sha256(config['data'][split]),
                       'doc_ids': [d['doc_id'] for d in docs], 'checkpoint': str(checkpoint),
                       'training_seed': config['training']['seed'], 'variant': model.variant})
        path = directory / ('%s_%02d_seed%d.json' % (condition, round(rate * 100), seed))
        write_json(path, result)
        rows.append({'condition': condition, 'rate': rate, 'noise_seed': seed,
                     'training_seed': config['training']['seed'], 'f1': result['metrics']['f1'],
                     'ign_f1': result['metrics']['ign_f1'], 'file': str(path)})

    summary = []
    for condition, rate in sorted(set((r['condition'], r['rate']) for r in rows)):
        subset = [r for r in rows if r['condition'] == condition and r['rate'] == rate]
        summary.append({'condition': condition, 'rate': rate, 'noise_manifests': len(subset),
                        'training_seeds': 1,
                        'f1_mean_over_noise': float(np.mean([r['f1'] for r in subset])),
                        'ign_f1_mean_over_noise': float(np.mean([r['ign_f1'] for r in subset]))})

    result = {'rows': rows, 'summary': summary,
              'scope': '同一 checkpoint 的噪声曲线；多个扰动种子不等于独立训练种子'}
    write_json(directory / 'summary.json', result)
    return result


@torch.no_grad()
def diagnose(checkpoint, split, output, device='auto', limit=10):
    if checkpoint_variant(checkpoint) == 'E1':
        from .etmier_engine import diagnose as diagnose_etmier
        return diagnose_etmier(checkpoint, split, output, device, limit)

    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs, encoded, _ = prepare(config, split, vocab, limit)
    store = TeacherStore(config, model.variant in TEACHER_VARIANTS)
    rows = []
    for doc, item in zip(docs, encoded):
        obs = observation_for(doc, 'clean', 0.0, 101, store, split)
        q = store.get(split, doc['doc_id'])['q_doc'] if model.requires_teacher else None
        inputs, _ = tensorize(item, obs, q, device)
        clean = model(inputs)
        disabled = model(inputs, force_gate=0.0)
        altered = dict(inputs)
        altered['observed_types'] = (inputs['observed_types'] + 1) % 6
        disabled_altered = model(altered, force_gate=0.0)
        text_error = float((disabled['logits'] - disabled['text_logits']).abs().max()) if disabled['logits'].numel() else 0.0
        invariance_error = float((disabled['logits'] - disabled_altered['logits']).abs().max()) if disabled['logits'].numel() else 0.0
        single = dict(inputs)
        single['observed_types'] = inputs['observed_types'].clone()
        single['observed_types'][0] = (single['observed_types'][0] + 1) % 6
        changed = model(single)
        per_pair_delta = (changed['logits'] - clean['logits']).abs().mean(dim=-1).cpu()
        adjacency = inputs['adjacency'].cpu() > 0
        distances = {0: 0}
        frontier = {0}
        for depth in (1, 2):
            next_frontier = {v for u in frontier for v in range(len(adjacency)) if adjacency[v, u]} - distances.keys()
            distances.update({v: depth for v in next_frontier})
            frontier = next_frontier

        grouped = {}
        for depth in (0, 1, 2, 3):
            values = [float(delta) for (h, t), delta in zip(doc['pairs'], per_pair_delta)
                      if min(distances.get(h, 3), distances.get(t, 3)) == depth]
            grouped[str(depth)] = {'pairs': len(values), 'mean_logit_change': float(np.mean(values)) if values else None}
        rows.append({'doc_id': doc['doc_id'], 'gate_off_text_max_error': text_error,
                     'gate_off_type_change_max_error': invariance_error,
                     'single_entity_type_change': {'entity_id': 0, 'pair_endpoint_min_distance': grouped}})

    passed = all(r['gate_off_text_max_error'] <= 1e-6 and r['gate_off_type_change_max_error'] <= 1e-6 for r in rows)
    result = {'passed': passed, 'tolerance': 1e-6, 'documents': rows,
              'scope': '同一模型的类型旁路检查与传播路径敏感性；不是因果证明或事实错误率'}
    write_json(output, result)
    if not passed:
        raise AssertionError('类型关闭后仍有泄漏，详细结果：' + str(output))
    return result


@torch.no_grad()
def evidence_sensitivity(checkpoint, split, output, device='auto', limit=100, source='predicted'):
    if checkpoint_variant(checkpoint) == 'E1':
        raise ValueError('当前 C5 接口适用于 B0–B8；E1 的证据注意力与 NA 决策分数需要独立适配。')

    if split == 'train':
        raise ValueError('C5 请使用 dev/test；当前重新编码接口使用最终类型头，不适用于 train OOF。')

    model, config, vocab, device = load_checkpoint(checkpoint, device)
    docs, encoded, _ = prepare(config, split, vocab)
    store = TeacherStore(config, model.requires_teacher)
    tokenizer = create_tokenizer(config)
    predictor = None
    if model.requires_teacher:
        from .teachers import TeacherPredictor
        predictor = TeacherPredictor(config, store.directory, device=str(device))

    rows = []
    for doc, item in zip(docs, encoded):
        obs = observation_for(doc, 'clean', 0.0, 101, store, split)
        q = store.get(split, doc['doc_id'])['q_doc'] if model.requires_teacher else None
        inputs, targets = tensorize(item, obs, q, device)
        baseline = model(inputs)

        for pair_index in torch.where(targets['evidence_mask'])[0].tolist():
            reference = torch.where(targets['evidence_labels'][pair_index] > 0)[0].tolist()
            if source == 'reference':
                selected = reference
            else:
                selected = baseline['evidence_logits'][pair_index].topk(min(2, len(doc['sents']))).indices.tolist()

            alternatives = [s for s in range(len(doc['sents'])) if s not in set(selected) | set(reference)]
            if len(alternatives) < len(selected):
                continue

            controls = []
            for sid in selected:
                match = min(alternatives, key=lambda candidate: abs(len(doc['sents'][candidate]) - len(doc['sents'][sid])))
                controls.append(match)
                alternatives.remove(match)

            relation_ids = torch.where(targets['relation_labels'][pair_index] > 0)[0]
            base_score = float(baseline['logits'][pair_index, relation_ids].sigmoid().mean())
            changed_scores = {}
            for action, chosen, keep in (('keep_evidence', selected, True), ('mask_evidence', selected, False),
                                         ('keep_control', controls, True), ('mask_control', controls, False)):
                changed = normalize_document(mask_evidence_sentences(doc, chosen, keep_only=keep), doc['doc_id'])
                changed_encoded = encode_document(changed, tokenizer, vocab, config['data']['max_length'], config['data']['overlap'])
                # 文本被改动后重编码全文类型器，外部观测类型仍保持原值
                changed_q = predictor.predict_document(changed)['q_doc'] if predictor is not None else None
                changed_inputs, _ = tensorize(changed_encoded, obs, changed_q, device)
                changed_output = model(changed_inputs)
                changed_scores[action] = float(changed_output['logits'][pair_index, relation_ids].sigmoid().mean())

            rows.append({'doc_id': doc['doc_id'], 'pair': doc['pairs'][pair_index],
                         'selected_sentences': selected, 'control_sentences': controls,
                         'selected_words': sum(len(doc['sents'][s]) for s in selected),
                         'control_words': sum(len(doc['sents'][s]) for s in controls),
                         'base_probability': base_score, 'modified_probabilities': changed_scores,
                         'comprehensiveness': base_score - changed_scores['mask_evidence'],
                         'sufficiency_drop': base_score - changed_scores['keep_evidence']})

            if len(rows) >= limit:
                break

        if len(rows) >= limit:
            break

    result = {'source': source, 'eligible_examples': len(rows), 'rows': rows,
              'scope': '仅实体对级有标注证据子集的敏感性；实体提及保留，参考标签不用于计算改文后的 F1'}
    write_json(output, result)
    return result
