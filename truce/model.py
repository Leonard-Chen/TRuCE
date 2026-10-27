#!/usr/bin/env python3
'''文档级关系抽取底座，含 B0-B8 干预配置'''

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 本项目只运行 PyTorch，避免 Transformers 自动加载环境中无关的 TensorFlow/JAX。
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')


class TinyEncoder(nn.Module):
    '''
    随机初始化的小编码器，仅用于离线流程验证，不作为 BERT 基线
    '''

    def __init__(self, hidden_size: int, vocab_size: int, max_length: int, dropout: float):
        super().__init__()
        heads = next(n for n in (4, 2, 1) if hidden_size % n == 0)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 2, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.max_length = max_length

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.shape[1] > self.max_length:
            raise ValueError('分块长度超过 tiny 编码器位置容量。')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.word_embeddings(input_ids) + self.position_embeddings(positions)[None]
        return self.layers(hidden, src_key_padding_mask=~attention_mask.bool())


class DocREModel(nn.Module):
    '''
    ★核心模型
    固定文本图、实体对证据池化及独立类型残差头
    '''

    NUM_TYPES = 6
    VARIANTS = {'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'}

    def __init__(self, config: Dict[str, Any], num_relations: int):
        super().__init__()
        cfg = config.get('model', config)
        self.variant = str(cfg.get('variant', 'B1')).upper()
        if self.variant not in self.VARIANTS:
            raise ValueError('variant 必须为 B0--B8；官方 ET-MIER 请走独立适配。')
        if num_relations < 1:
            raise ValueError('num_relations 应仅包含至少一个正关系类别，不含 NA。')
        self.num_relations = num_relations
        self.hidden_size = int(cfg.get('hidden_size', 128))
        self.graph_layers = int(cfg.get('graph_layers', 2))
        self.pair_chunk_size = int(cfg.get('pair_chunk_size', 256))
        self.encoder_batch_size = int(cfg.get('encoder_batch_size', 1))
        self.type_weight = float(cfg.get('type_weight', 0.5))
        self.type_dropout = float(cfg.get('type_dropout', 0.2))
        dropout = float(cfg.get('dropout', 0.1))
        if min(self.hidden_size, self.graph_layers, self.pair_chunk_size, self.encoder_batch_size) < 1:
            raise ValueError('隐藏维度、图层数和分块 batch 必须为正整数。')
        if not 0 <= self.type_dropout <= 1 or not 0 <= dropout < 1:
            raise ValueError('dropout 参数范围不合法。')
        self.backend = str(cfg.get('encoder_backend', 'hf'))
        if self.backend == 'tiny':
            self.encoder_hidden_size = int(cfg.get('tiny_hidden_size', self.hidden_size))
            self.encoder = TinyEncoder(
                self.encoder_hidden_size, int(cfg.get('tiny_vocab_size', cfg.get('vocab_size', 4096))),
                int(cfg.get('max_length', config.get('data', {}).get('max_length', 512))), dropout,
            )
        elif self.backend == 'hf':
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(
                str(cfg.get('encoder_name', 'bert-base-uncased')),
                revision=cfg.get('encoder_revision', 'main'),
                local_files_only=bool(cfg.get('local_files_only', False)),
            )
            self.encoder_hidden_size = int(self.encoder.config.hidden_size)
            if bool(cfg.get('gradient_checkpointing', False)):
                self.encoder.gradient_checkpointing_enable()
        else:
            raise ValueError('encoder_backend 只能是 hf 或 tiny。')
        self.encoder_projection = (
            nn.Identity() if self.encoder_hidden_size == self.hidden_size
            else nn.Linear(self.encoder_hidden_size, self.hidden_size)
        )
        h = self.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.text_graph = nn.ModuleList(nn.Linear(h, h) for _ in range(self.graph_layers))
        self.text_norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(self.graph_layers))
        self.pair_projection = nn.Linear(4 * h, h)
        self.evidence_query = nn.Linear(h, h, bias=False)
        self.evidence_key = nn.Linear(h, h, bias=False)
        self.text_classifier = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, num_relations))

        # 类型通道没有 bias/平移归一化/旁路，保证门控全关时能严格回到本模型文本分支
        self.type_embedding = nn.Parameter(torch.empty(self.NUM_TYPES, h))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.type_graph = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(self.graph_layers))
        self.type_classifier = nn.Sequential(
            nn.Linear(6 * h, h, bias=False),
            nn.GELU(),
            nn.Linear(h, num_relations, bias=False)
        )

        self.uses_c = self.variant in {'B6', 'B8'}

        self.uses_g = self.variant in {'B5', 'B7', 'B8'}

        self.requires_teacher = self.variant in {'B4', 'B5', 'B6', 'B7', 'B8'}

        # C：软纠正模块
        self.corrector = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        ) if self.uses_c else None
        # G：传播前抑制模块
        self.gate_network = nn.Sequential(
            nn.Linear(h + 5, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        ) if self.uses_g else None

        if self.variant == 'B0':
            self.type_embedding.requires_grad_(False)
            self.type_graph.requires_grad_(False)
            self.type_classifier.requires_grad_(False)

    def encode_tokens(self, inputs: Dict[str, Any], project: bool = True) -> Tensor:
        '''按全局 wordpiece 索引合并重叠分块；project=False 返回原编码器维度。'''
        count = int(inputs['num_tokens'])
        if count <= 0:
            raise ValueError('num_tokens 必须大于 0，不能编码空文档。')

        ids, mask, mapping = inputs['input_ids'], inputs['attention_mask'], inputs['token_map']
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != mapping.shape:
            raise ValueError('input_ids、attention_mask、token_map 必须是形状相同的二维张量。')

        merged = None
        coverage = torch.zeros(count, device=ids.device, dtype=torch.long)
        for start in range(0, ids.shape[0], self.encoder_batch_size):
            end = start + self.encoder_batch_size

            if self.backend == 'tiny':
                encoded = self.encoder(ids[start:end], mask[start:end])
            else:
                encoded = self.encoder(input_ids=ids[start:end], attention_mask=mask[start:end]).last_hidden_state

            positions = mapping[start:end].reshape(-1)
            valid = (positions >= 0) & mask[start:end].reshape(-1).bool()
            positions = positions[valid]
            if positions.numel() and int(positions.max()) >= count:
                raise ValueError('token_map 超过 num_tokens 指定的全局范围。')

            values = encoded.reshape(-1, self.encoder_hidden_size)[valid]
            if merged is None:
                merged = encoded.new_zeros((count, self.encoder_hidden_size))
            merged = merged.index_add(0, positions, values)
            coverage.index_add_(0, positions, torch.ones_like(positions))

        if merged is None or bool((coverage == 0).any()):
            raise ValueError('分块没有覆盖全部正文 token，禁止静默截断长文档。')

        merged = merged / coverage.to(merged.dtype).unsqueeze(-1)
        return self.encoder_projection(merged) if project else merged

    @staticmethod
    def _pool_spans(tokens: Tensor, groups: Sequence[Sequence[Tuple[int, int]]]) -> Tensor:
        representations: List[Tensor] = []

        for spans in groups:
            if not spans:
                raise ValueError('实体/句子池化区间不能为空。')

            for begin, end in spans:
                if not (0 <= begin < end <= len(tokens)):
                    raise ValueError('实体或句子位置超出全文 wordpiece 范围。')
            representations.append(torch.stack([tokens[begin:end].mean(0) for begin, end in spans]).mean(0))

        if not representations:
            return tokens.new_empty((0, tokens.shape[-1]))

        return torch.stack(representations)

    def _type_inputs(self, inputs: Dict[str, Any], entities: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        observed = inputs['observed_types']
        n = len(entities)
        if observed.shape != (n,) or bool(((observed < -1) | (observed >= self.NUM_TYPES)).any()):
            raise ValueError('observed_types 应为每实体一个 -1 或 0--5 类型索引。')

        missing = observed < 0
        p_obs = F.one_hot(observed.clamp_min(0), self.NUM_TYPES).float()
        p_obs = p_obs * (~missing).float().unsqueeze(-1)
        ones = entities.new_ones(n)
        zeros = entities.new_zeros(n)
        if not self.requires_teacher:
            return (torch.zeros_like(p_obs) if self.variant == 'B0' else p_obs), zeros, ones

        q_doc = inputs.get('q_doc')
        if q_doc is None:
            raise ValueError('%s 必须提供独立冻结 q_doc 缓存；不能用均匀概率冒充类型器。' % self.variant)

        # 熵与概率混合保持 FP32，避免 AMP 将极小概率变成 0 后产生 0×log(0)。
        q = q_doc.detach().to(device=entities.device, dtype=torch.float32)
        if q.shape != (n, self.NUM_TYPES) or not bool(torch.isfinite(q).all()) or bool((q < 0).any()) or bool((q.sum(-1) <= 0).any()):
            raise ValueError('q_doc 必须是形状 [实体数, 6] 的有效概率分布。')

        q = q / q.sum(-1, keepdim=True)
        if self.variant == 'B4':
            return q, ones, ones

        support = (p_obs * q).sum(-1)
        confidence = q.max(-1).values
        entropy = -(q * q.clamp_min(1e-8).log()).sum(-1) / math.log(self.NUM_TYPES)
        reliability = torch.stack((support, confidence, entropy, missing.to(q.dtype)), dim=-1)
        alpha = torch.sigmoid(self.corrector(reliability)).squeeze(-1) if self.corrector is not None else zeros
        corrected = (1 - alpha[:, None]) * p_obs + alpha[:, None] * q if self.uses_c else p_obs
        if self.uses_c:
            corrected = torch.where(missing[:, None], q, corrected)
            alpha = torch.where(missing, ones, alpha)

        gates = ones
        if self.gate_network is not None:
            agreement = (corrected * q).sum(-1, keepdim=True)
            gate_input = torch.cat((entities, reliability, agreement), dim=-1)
            gates = torch.sigmoid(self.gate_network(gate_input)).squeeze(-1)

        return corrected, alpha, gates

    def forward(self, inputs: Dict[str, Any], force_gate: Optional[Any] = None) -> Dict[str, Any]:
        '''
        前向传播
        '''
        tokens = self.encode_tokens(inputs)
        entity_base = self._pool_spans(tokens, inputs['entity_spans'])
        sentences = self._pool_spans(tokens, [[span] for span in inputs['sentence_spans']])

        n, h = entity_base.shape
        if n < 1 or len(sentences) < 1:
            raise ValueError('文档必须至少有一个实体和一个句子。')

        adjacency = inputs['adjacency'].to(device=tokens.device, dtype=tokens.dtype)
        if adjacency.shape != (n, n) or not bool(torch.isfinite(adjacency).all()) or bool((adjacency < 0).any()):
            raise ValueError('adjacency 必须是非负有限的 [实体数, 实体数] 固定邻接矩阵。')

        if not torch.allclose(adjacency.sum(-1).float(), torch.ones(n, device=tokens.device), atol=1e-3, rtol=1e-3):
            raise ValueError('adjacency 应预先按行归一化，并包含自环。')

        pairs = inputs['pairs']
        if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.numel() and bool(((pairs < 0) | (pairs >= n)).any())):
            raise ValueError('pairs 必须是合法的 [实体对数, 2] 索引。')

        entities = entity_base
        for layer, norm in zip(self.text_graph, self.text_norms):
            entities = norm(entities + self.dropout(F.gelu(adjacency @ layer(entities))))

        probs, alpha, gates = self._type_inputs(inputs, entity_base)
        if force_gate is not None:
            gates = torch.as_tensor(force_gate, dtype=tokens.dtype, device=tokens.device)
            if gates.ndim == 0:
                gates = gates.expand(n)
            if gates.shape != (n,) or not bool(torch.isfinite(gates).all()) or bool(((gates < 0) | (gates > 1)).any()):
                raise ValueError('force_gate 必须为 0-1 标量或每实体一个 0-1 值。')

        type_states = probs @ self.type_embedding
        if self.variant == 'B2' and self.training:
            keep = torch.rand(n, device=tokens.device) >= self.type_dropout
            type_states = type_states * keep.to(type_states.dtype)[:, None]

        if self.variant == 'B0':
            type_states = torch.zeros_like(type_states)

        for layer in self.type_graph:
            # 先抑制来源再按原图聚合；含自环，不要把乘 gate 以后的权重又重新归一，否则 gate 会失效
            source = type_states if self.variant == 'B5' else gates[:, None] * type_states
            type_states = self.dropout(F.gelu(adjacency @ layer(source)))

        if self.variant == 'B5':
            type_states = gates[:, None] * type_states

        evidence_keys = self.evidence_key(sentences)
        text_parts, type_parts, evidence_parts = [], [], []
        for offset in range(0, len(pairs), self.pair_chunk_size):
            selected = pairs[offset:offset + self.pair_chunk_size]
            head, tail = entities[selected[:, 0]], entities[selected[:, 1]]
            pair_state = F.gelu(self.pair_projection(torch.cat((head, tail, head * tail, (head - tail).abs()), -1)))
            # 实体对分块只建立 [pair, sentence] 注意力，不展开 [pair, sentence, hidden]
            evidence_logits = self.evidence_query(pair_state) @ evidence_keys.t() / math.sqrt(h)
            context = evidence_logits.softmax(-1) @ sentences
            text_logits = self.text_classifier(torch.cat((pair_state, context), -1))
            th, tt = type_states[selected[:, 0]], type_states[selected[:, 1]]
            type_features = torch.cat((th, tt, th * tt, (th - tt).abs(), th * head, tt * tail), -1)
            type_logits = self.type_classifier(type_features)

            if self.variant == 'B3':
                type_logits = self.type_weight * type_logits
            text_parts.append(text_logits)
            type_parts.append(type_logits)
            evidence_parts.append(evidence_logits)

        connected_zero = tokens.sum() * 0
        text_logits = torch.cat(text_parts) if text_parts else tokens.new_empty((0, self.num_relations)) + connected_zero
        type_logits = torch.cat(type_parts) if type_parts else tokens.new_empty((0, self.num_relations)) + connected_zero
        evidence_logits = torch.cat(evidence_parts) if evidence_parts else tokens.new_empty((0, len(sentences))) + connected_zero

        return {
            'logits': text_logits + type_logits,
            'text_logits': text_logits,
            'type_logits': type_logits,
            'evidence_logits': evidence_logits,
            'corrected_probs': probs,
            'correction_alpha': alpha,
            'gates': gates,
            'uses_c': self.uses_c,
            'diagnostics': {
                'mean_gate': gates.mean().detach(),
                'type_residual_norm': type_logits.norm().detach(),
                'encoder_backend': self.backend,
                'type_node_states': type_states.detach(),
            },
        }
