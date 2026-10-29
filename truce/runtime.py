#!/usr/bin/env python3
import copy
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    # 逐块写入大型干预清单，避免同时保留完整 JSON 字符串及其编码副本。
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def deep_update(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_path(value):
    value = os.fspath(value)

    def replace(match):
        name = next(group for group in match.groups() if group is not None)
        if name not in os.environ:
            raise ValueError('路径引用了未设置的环境变量：' + name)
        return os.environ[name]

    value = re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%', replace, value)
    return os.path.expanduser(value)


def resolve_path(value):
    expanded = _expand_path(value)
    windows = PureWindowsPath(expanded)
    if os.name != 'nt' and (windows.drive or expanded.startswith('\\\\')):
        raise ValueError('当前系统不能使用 Windows 路径，请通过 DOCRE_* 环境变量或 --set 改为服务器路径：' + expanded)
    path = Path(expanded)
    return path if path.is_absolute() else ROOT / path


def _read_config(path, ancestors=()):
    path = Path(_expand_path(path)).resolve()
    if path in ancestors:
        raise ValueError('配置文件存在循环继承：' + str(path))
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig')) or {}
    if not isinstance(config, dict):
        raise ValueError('配置文件顶层必须是字典：' + str(path))
    parent = config.pop('extends', None)
    if parent:
        config = deep_update(_read_config(path.parent / _expand_path(parent), ancestors + (path,)), config)
    config['config_path'] = str(path)
    return config


def relocate_config(config, overrides=None):
    config = copy.deepcopy(config)
    data_directory = os.environ.get('DOCRE_DATA_DIR')
    if data_directory:
        for key in ('train', 'dev', 'test'):
            value = config.get('data', {}).get(key)
            if value:
                # 迁移旧 checkpoint 时同时识别 Windows 和 Linux 的文件名分隔符。
                filename = str(value).replace('\\', '/').rsplit('/', 1)[-1]
                config['data'][key] = str(resolve_path(data_directory) / filename)
    for variable, section, field in (
        ('DOCRE_TEACHER_DIR', 'teacher', 'cache_dir'),
        ('DOCRE_OUTPUT_DIR', 'output', 'directory'),
        ('DOCRE_ETMIER_DIR', 'etmier', 'source_dir'),
        ('DOCRE_ENCODER_NAME', 'model', 'encoder_name'),
    ):
        if os.environ.get(variable):
            config.setdefault(section, {})[field] = os.environ[variable]
    # 显式参数优先于服务器环境变量，避免 --set 被环境中残留的设置覆盖。
    for override in overrides or []:
        dotted, separator, value = override.partition('=')
        if not separator or not dotted or any(not field for field in dotted.split('.')):
            raise ValueError('参数覆盖必须使用 key=value：' + override)
        current = config
        fields = dotted.split('.')
        for field in fields[:-1]:
            current = current.setdefault(field, {})
        current[fields[-1]] = yaml.safe_load(value)
    paths = [('data', name) for name in ('train', 'dev', 'test')]
    paths += [('teacher', 'cache_dir'), ('output', 'directory'), ('etmier', 'source_dir')]
    for section, key in paths:
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = str(resolve_path(value))
    encoder = config.get('model', {}).get('encoder_name')
    if encoder:
        expanded = _expand_path(encoder)
        local = (expanded.startswith(('.', '/', '\\')) or bool(PureWindowsPath(expanded).drive)
                 or (ROOT / expanded).is_dir())
        config['model']['encoder_name'] = str(resolve_path(expanded)) if local else expanded
    return config


def load_config(path, overrides=None):
    return relocate_config(_read_config(path), overrides)


def resolve_resume_path(value):
    path = resolve_path(value).resolve()
    if path.name == 'training.pt':
        run_directory = path.parent.parent
        state_path = path
    elif path.name == 'last':
        run_directory = path.parent
        state_path = path / 'training.pt'
    else:
        run_directory = path
        state_path = path / 'last' / 'training.pt'
    if not state_path.is_file():
        raise FileNotFoundError('缺少完整训练状态，只有模型权重不能断点续训：' + str(state_path))
    if not (run_directory / 'run.json').is_file():
        raise FileNotFoundError('断点目录缺少 run.json：' + str(run_directory))
    return run_directory, state_path


def load_resume_config(value, config_path=None, overrides=None):
    run_directory, _ = resolve_resume_path(value)
    if config_path is not None:
        return load_config(config_path, overrides)
    metadata = read_json(run_directory / 'run.json')
    if not isinstance(metadata.get('config'), dict):
        raise ValueError('run.json 中缺少原训练配置，不能按默认 B0 继续训练。')
    return relocate_config(metadata['config'], overrides)


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(value='auto'):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('当前 PyTorch 不支持 CUDA；请使用 --device cpu 或安装 GPU 版 PyTorch。')
    return torch.device(value)


def load_split(path, limit=None):
    from .data import load_documents, normalize_document
    documents = load_documents(path)
    if limit:
        documents = documents[:limit]
    result, seen = [], set()
    for raw in documents:
        doc_id = raw.get('title')
        if not doc_id or doc_id in seen:
            raise ValueError('文档 title 缺失或重复，不能建立稳定文档身份。')
        seen.add(doc_id)
        result.append(normalize_document(raw, doc_id))
    return result


def tensorize(encoded, observation, q_doc=None, device='cpu'):
    chunks = encoded['chunks']
    length = max(len(chunk['input_ids']) for chunk in chunks)
    inputs = {}
    for key, padding in (('input_ids', 0), ('attention_mask', 0), ('token_map', -1)):
        inputs[key] = torch.tensor([
            chunk[key] + [padding] * (length - len(chunk[key])) for chunk in chunks
        ], dtype=torch.long, device=device)
    inputs.update({
        'num_tokens': encoded['num_tokens'],
        'entity_spans': encoded['entity_spans'],
        'sentence_spans': encoded['sentence_spans'],
        'adjacency': torch.tensor(encoded['adjacency'], dtype=torch.float32, device=device),
        'pairs': torch.tensor(encoded['pairs'], dtype=torch.long, device=device).reshape(-1, 2),
        'observed_types': torch.tensor(observation['observed_types'], dtype=torch.long, device=device),
    })
    if q_doc is not None:
        inputs['q_doc'] = torch.tensor(q_doc, dtype=torch.float32, device=device)
    metadata = encoded.get('targets', {})
    targets = {
        'relation_labels': torch.tensor(encoded['relation_targets'], dtype=torch.float32, device=device).reshape(
            len(encoded['pairs']), encoded['num_relations']),
        'evidence_labels': torch.tensor(encoded['evidence_targets'], dtype=torch.float32, device=device).reshape(
            len(encoded['pairs']), len(encoded['sentence_spans'])),
        'evidence_mask': torch.tensor(encoded['evidence_mask'], dtype=torch.bool, device=device),
        'reference_types': torch.tensor(metadata['reference_types'], dtype=torch.long, device=device),
        'type_consistent': torch.tensor(metadata['class_consistent'], dtype=torch.bool, device=device),
    }
    return inputs, targets


def encode_split(documents, tokenizer, vocab, config):
    from .data import encode_document
    data = config['data']
    return [encode_document(doc, tokenizer, vocab, data.get('max_length', 512),
                            data.get('overlap', 128)) for doc in documents]


def environment_info():
    import transformers
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        'python': platform.python_version(), 'python_executable': sys.executable,
        'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'numpy': np.__version__, 'cuda_available': torch.cuda.is_available(),
        'cuda_runtime': torch.version.cuda, 'git_commit': commit,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'pid': os.getpid(),
    }


def source_manifest():
    files = list(Path(__file__).resolve().parent.glob('*.py')) + list((ROOT / 'configs').glob('*.yaml'))
    files += list((ROOT / 'scripts').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(files)}
