#!/usr/bin/env python3
'''多标签关系、可用证据和一致参考类型的训练目标'''

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_loss(outputs: Dict[str, Any], targets: Dict[str, Tensor], weights: Optional[Dict[str, float]] = None) -> Dict[str, Tensor]:
    '''
    损失函数
    '''
    weights = weights or {}
    logits = outputs['logits']
    relation_labels = targets.get('relation_labels', targets.get('relation_targets'))
    if relation_labels is None or relation_labels.shape != logits.shape:
        raise ValueError('relation_labels 形状须与正关系 logits 一致，不应包含 NA 列。')

    relation = F.binary_cross_entropy_with_logits(logits, relation_labels.to(logits)) if logits.numel() else logits.sum() * 0
    evidence_logits = outputs['evidence_logits']
    evidence = evidence_logits.sum() * 0
    evidence_labels = targets.get('evidence_labels', targets.get('evidence_targets'))
    if evidence_labels is not None:
        if evidence_labels.shape != evidence_logits.shape:
            raise ValueError('evidence_labels 与 evidence_logits 形状不一致。')

        labels = evidence_labels.to(evidence_logits)
        valid = labels.sum(-1) > 0
        if targets.get('evidence_mask') is not None:
            valid = valid & targets['evidence_mask'].to(device=valid.device, dtype=torch.bool)

        # 空 evidence 表示缺标，不能作为全负证据；只监督标注非空实体对的句子分布
        if bool(valid.any()):
            distribution = labels[valid] / labels[valid].sum(-1, keepdim=True)
            evidence = F.kl_div(F.log_softmax(evidence_logits[valid], dim=-1), distribution, reduction='batchmean')

    type_loss = outputs['corrected_probs'].sum() * 0
    reference = targets.get('reference_types')
    consistent = targets.get('type_consistent', targets.get('class_consistent'))
    if outputs.get('uses_c', False) and reference is not None:

        probabilities = outputs['corrected_probs']
        reference = reference.to(device=probabilities.device, dtype=torch.long)
        valid = (reference >= 0) & (reference < probabilities.shape[-1])

        if consistent is not None:
            valid = valid & consistent.to(device=valid.device, dtype=torch.bool)

        if bool(valid.any()):
            type_loss = F.nll_loss(probabilities[valid].clamp_min(1e-8).log(), reference[valid])

    total = (
        float(weights.get('relation', 1.0)) * relation
        + float(weights.get('evidence', 1.0)) * evidence
        + float(weights.get('type', 0.1)) * type_loss
    )
    return {'total': total, 'relation': relation, 'evidence': evidence, 'type': type_loss}
