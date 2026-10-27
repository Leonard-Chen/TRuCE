#!/usr/bin/env python3
'''ET-MIER 官方模型的独立兼容层；类型标注只参与辅助训练监督'''

import builtins
import hashlib
import math
import os
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import resolve_path


os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')


def _source_directory(value: str) -> Path:
    path = resolve_path(value).resolve()
    if (path / 'ET-MIER-ReDocRED').is_dir():
        path = path / 'ET-MIER-ReDocRED'
    if not (path / 'model_SAIS_Evidence.py').is_file():
        raise FileNotFoundError('未找到 ET-MIER-ReDocRED/model_SAIS_Evidence.py：{}'.format(path))
    return path


def _source_commit(source: Path, expected=None):
    try:
        actual = subprocess.run(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], check=True,
            capture_output=True, text=True, encoding='utf-8',
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if expected:
            raise RuntimeError('无法核验 ET-MIER source_commit；请使用固定版本的 Git checkout。') from error
        return None
    if expected and actual.lower() != str(expected).lower():
        raise RuntimeError('ET-MIER source_commit 不匹配：要求 {}，实际 {}'.format(expected, actual))
    changes = subprocess.run(
        ['git', '-C', str(source), 'diff', 'HEAD', '--quiet'],
        capture_output=True, text=True, encoding='utf-8',
    )
    if changes.returncode == 1:
        raise RuntimeError('ET-MIER 已跟踪源码存在未提交改动，不能视为固定 source_commit。')
    if changes.returncode:
        raise RuntimeError('无法核验 ET-MIER 已跟踪源码是否与 source_commit 一致。')
    return actual


def _load_official(source: Path, hidden_size: int, settings: Dict[str, Any]):
    '''在独立命名空间加载原始源文件，避免原仓库解析本项目命令行和写日志。'''
    constants = types.ModuleType('constant_SAIS_Evidence')
    values = {
        'torch': torch, 'tau': 2.0, 'tau_base': 1.0, 'lambda_3': 1.0,
        'Crht_loss': False, 'Emb_loss': False, 'Relation_loss': True,
        'Relation_Seploss': True, 'Relation_loss_lambda': float(settings.get('relation_loss_lambda', 0.1)),
        'PEMSCLloss': False, 'TASK_ET': True, 'Top_k': int(settings.get('top_k', 1)),
        'loss_weight_ET': float(settings.get('type_loss_weight', 0.25)),
        'TSER_Logsumexp': bool(settings.get('tser_logsumexp', False)),
        'enhance_evi': ['test', 'dev', 'train'] if settings.get('enhance_evidence', False) else [],
        's_attn_lambda': float(settings.get('s_attn_lambda', 0.0)),
        'one_mention_copy_or_addrel': False, 'three_atten': bool(settings.get('three_atten', False)),
        'use_cls_info': False, 'use_bilinear': True, 'Relation_Specific_num': 6,
        'reduced_dim': hidden_size, 'relation_dim': hidden_size, 'extractor_dim': hidden_size,
    }
    values['s_attn_logsumexp_lambda'] = 1.0 - values['s_attn_lambda']
    constants.__dict__.update(values)
    loaded = {'constant_SAIS_Evidence': constants}
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in loaded:
            return loaded[name]
        return original_import(name, globals, locals, fromlist, level)

    hashes = {}
    patches = []
    names = ('long_seq', 'losses_SAIS_Evidence', 'Model_assistant_SAIS_Evidence', 'model_SAIS_Evidence')
    for name in names:
        path = source / (name + '.py')
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        code = raw.decode('utf-8-sig').replace('\r\n', '\n')
        if name == 'model_SAIS_Evidence':
            changes = {
                'from prepro import docred_ent2id\n': '',
                '        with open(\'env_sent.pkl\', \'wb\') as f:\n            pickle.dump([], f)\n': '',
            }
            for before, after in changes.items():
                if code.count(before) != 1:
                    raise RuntimeError('ET-MIER 源码已变化，需重新核验兼容补丁：{}'.format(before.strip()))
                code = code.replace(before, after)
                patches.append({'file': path.name, 'remove': before.strip()})
        module = types.ModuleType('_docre_etmier_' + name)
        module.__file__ = str(path)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=isolated_import)
        exec(compile(code, str(path), 'exec'), module.__dict__)
        loaded[name] = module
    for name in ('constant_SAIS_Evidence.py', 'args_iscf.py'):
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    return loaded['model_SAIS_Evidence'].DocREModel, values, hashes, patches


class ETMIERAdapter(nn.Module):
    '''保留官方关系计算与损失，显式分离观测类型和训练参考类型。'''

    def __init__(self, config: Dict[str, Any], num_relations: int, encoder=None, tokenizer=None):
        super().__init__()
        model_cfg = config.get('model', config)
        settings = config.get('etmier', {})
        self.num_relations = int(num_relations)
        self.variant = 'E1'
        if self.num_relations < 1:
            raise ValueError('num_relations 必须是不含 NA 的正关系类别数。')
        self.settings = dict(settings)
        backend = str(model_cfg.get('encoder_backend', 'hf'))
        if encoder is None:
            if backend == 'tiny':
                from transformers import BertConfig, BertModel
                hidden = int(model_cfg.get('tiny_hidden_size', 32))
                heads = next(h for h in (4, 2, 1) if hidden % h == 0)
                tiny_config = BertConfig(
                    vocab_size=int(model_cfg.get('tiny_vocab_size', 4096)), hidden_size=hidden,
                    num_hidden_layers=3 if settings.get('three_atten', False) else 1,
                    num_attention_heads=heads, intermediate_size=hidden * 2,
                    hidden_dropout_prob=float(model_cfg.get('dropout', 0.1)),
                    attention_probs_dropout_prob=float(model_cfg.get('dropout', 0.1)),
                    max_position_embeddings=512,
                )
                tiny_config._attn_implementation = 'eager'
                encoder = BertModel(tiny_config)
            elif backend == 'hf':
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained(
                    str(model_cfg.get('encoder_name', 'bert-base-uncased')),
                    revision=model_cfg.get('encoder_revision', 'main'),
                    local_files_only=bool(model_cfg.get('local_files_only', False)),
                    attn_implementation='eager',
                )
            else:
                raise ValueError('ET-MIER encoder_backend 仅支持 hf 或 tiny。')
        encoder.config._attn_implementation = 'eager'
        if bool(model_cfg.get('gradient_checkpointing', False)):
            encoder.gradient_checkpointing_enable()
        encoder_config = encoder.config
        transformer_type = str(getattr(encoder_config, 'model_type', 'bert'))
        if transformer_type not in ('bert', 'roberta'):
            raise ValueError('官方长文本实现仅支持 BERT/RoBERTa。')
        encoder_config.transformer_type = transformer_type
        encoder_config.cls_token_id = int(getattr(tokenizer, 'cls_token_id', 1) or 0)
        encoder_config.sep_token_id = int(getattr(tokenizer, 'sep_token_id', 2) or 0)
        encoder_config.num_labels = self.num_relations + 1
        hidden = int(encoder_config.hidden_size)
        block_size = int(settings.get('block_size', 8 if backend == 'tiny' else 64))
        if hidden % block_size:
            raise ValueError('ET-MIER hidden_size 必须能被 block_size 整除。')
        if not 1 <= int(settings.get('top_k', 1)) <= 7:
            raise ValueError('ET-MIER top_k 必须在 1..7。')
        source = _source_directory(settings.get('source_dir', 'external/ET-MIER'))
        source_commit = _source_commit(source, settings.get('source_commit'))
        official_class, constants, hashes, patches = _load_official(source, hidden, settings)

        class CapturedOfficial(official_class):
            def forward_rel(self, hs, ts, rs):
                logits, features = super().forward_rel(hs, ts, rs)
                self.captured_logits = logits
                return logits, features

            def get_hrt(self, *args, **kwargs):
                result = super().get_hrt(*args, **kwargs)
                self.captured_doc_attention = result[5].mean(1)
                self.captured_batch_rel = result[6]
                return result

            def encode(self, *args, **kwargs):
                result = super().encode(*args, **kwargs)
                self.captured_sequence, self.captured_attention = result
                return result

        self.official = CapturedOfficial(
            encoder_config, encoder, tokenizer, emb_size=hidden, block_size=block_size,
            num_labels=min(int(settings.get('num_labels', 4)), self.num_relations + 1),
            max_sent_num=int(settings.get('max_sent_num', 25)),
            evi_thresh=float(settings.get('evidence_threshold', 0.2)),
        )
        self.evidence_weight = float(settings.get('evidence_loss_weight', 0.1))
        self.enhance_evidence = bool(settings.get('enhance_evidence', False))
        self.s_attn_lambda = float(settings.get('s_attn_lambda', 0.0))
        self.max_input_length = 1022 if transformer_type == 'bert' else 1021
        self._reference_mask = None
        self.official.SAISLoss.cal_ET_loss = self._masked_type_loss
        adapter_files = ('etmier_model.py', 'etmier_data.py', 'tokenization.py')
        adapter_hashes = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in adapter_files
        }
        self.provenance = {
            'source_dir': str(source), 'source_commit': source_commit,
            'source_sha256': hashes, 'in_memory_patches': patches,
            'adapter_sha256': adapter_hashes,
            'architecture': 'official ET-MIER-ReDocRED DocREModel', 'encoder_hidden_size': hidden,
            'constants': {k: v for k, v in constants.items() if k != 'torch'},
            'observed_types_consumed_at_prediction': False,
            'type_targets': 'clean reference_epair_types; inconsistent clusters masked',
            'evaluation_tag': 'infer (deterministic; no upstream test-tag Bernoulli attention)',
            'evidence_loss': 'official KL restricted to positive pairs with nonempty annotation',
            'encoder_adaptation': 'reduced_dim/relation_dim/extractor_dim follow encoder hidden size',
            'tiny_is_pretrained': False if backend == 'tiny' else None,
        }

    @property
    def encoder(self):
        return self.official.model

    def _masked_type_loss(self, logits, labels):
        target = labels.T.flatten().long() - 1
        valid = (target >= 0) & (target < 6)
        if self._reference_mask is not None:
            valid = valid & self._reference_mask.T.flatten().bool()
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], target[valid])

    def forward(self, inputs: Dict[str, Any], targets: Optional[Dict[str, Any]] = None):
        input_ids = inputs['input_ids']
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_input_length:
            raise ValueError('ET-MIER 两段编码仅完整覆盖最多 {} 个输入 token；请显式处理超长文档。'.format(self.max_input_length))
        supplied = dict(inputs)
        if targets is not None:
            supplied.update(targets)
        train_loss = targets is not None
        labels = supplied.get('labels') if train_loss else None
        reference_types = supplied.get('reference_epair_types') if train_loss else None
        if train_loss and (labels is None or reference_types is None):
            raise ValueError('ET-MIER 训练必须提供 labels 和干净的 reference_epair_types。')
        if train_loss and labels.shape[-1] != self.num_relations + 1:
            raise ValueError('ET-MIER labels 第 0 列必须为 NA，其后才是正关系。')
        self._reference_mask = supplied.get('reference_epair_mask') if train_loss else None
        # 观测类型不进入原模型预测，也不替换辅助训练真值；C1 的缺失编码不会成为负索引。
        official_output = self.official(
            input_ids=input_ids, attention_mask=inputs['attention_mask'],
            entity_pos=inputs['entity_pos'], hts=inputs['hts'], sent_pos=inputs['sent_pos'],
            labels=labels.clone() if labels is not None else None,
            epair_types=reference_types, sent_labels=None, tag='train' if train_loss else 'infer',
        )
        logits = self.official.captured_logits
        batch_rel = self.official.captured_batch_rel
        evidence = self.official.forward_evi(
            self.official.captured_doc_attention, inputs['sent_pos'], batch_rel, 1,
        )
        if self.enhance_evidence:
            attn, rels = self.official.get_hrt_logsumexp(
                self.official.captured_sequence, self.official.captured_attention,
                inputs['entity_pos'], inputs['hts'], 1,
            )
            extra = self.official.forward_evi(attn, inputs['sent_pos'], rels, 1)
            evidence = self.s_attn_lambda * evidence + (1.0 - self.s_attn_lambda) * extra
        result = {
            'logits': logits, 'relation_margins': logits[:, 1:] - logits[:, :1],
            'rel_pred': official_output['rel_pred'], 'evidence_scores': evidence,
            'evi_pred': evidence > self.official.evi_thresh,
        }
        if train_loss:
            losses = official_output['loss']
            zero = logits.sum() * 0.0
            evidence_loss = zero
            sentence_labels = supplied.get('sent_labels')
            if sentence_labels is not None:
                if sentence_labels.shape != evidence.shape:
                    raise ValueError('sent_labels 必须与实体对及 batch 句子 padding 对齐。')
                # 空 evidence 是未标注，不能当成全负证据；关系监督仍覆盖这些实体对。
                valid = (labels[:, 1:].sum(-1) > 0) & (sentence_labels.sum(-1) > 0)
                if supplied.get('evidence_mask') is not None:
                    valid = valid & supplied['evidence_mask'].bool()
                if bool(valid.any()):
                    distribution = sentence_labels[valid].float()
                    distribution = distribution / distribution.sum(-1, keepdim=True).clamp_min(1e-30)
                    evidence_loss = F.kl_div(
                        evidence[valid].float().clamp_min(1e-30).log(), distribution.clamp_min(1e-30), reduction='batchmean',
                    )
            components = {
                'relation': losses['rel_loss'], 'evidence': evidence_loss * self.evidence_weight,
                'type': losses.get('etoss', zero), 'prototype': losses.get('relation_seploss', zero),
            }
            components['total'] = sum(components.values())
            result['loss'] = components
        return result
