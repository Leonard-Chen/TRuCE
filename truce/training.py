#!/usr/bin/env python3
'''训练模块'''

import copy
import json
import math
import os
import random
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .runtime import resolve_resume_path, write_json
from .training_log import TrainingLogger


def capture_rng(device=None):
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                  numpy_state[2], numpy_state[3], numpy_state[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if device is not None and torch.device(device).type == 'cuda':
        state['active_cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device=None):
    random.setstate(state['python'])
    name, values, position, has_gauss, cached_gauss = state['numpy']
    np.random.set_state((name, values.numpy().astype(np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        for index, value in enumerate(state['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, index)
        # 迁移服务器后可见 GPU 编号可能改变，将训练设备的随机状态映射到新设备
        if device is not None and torch.device(device).type == 'cuda' and 'active_cuda' in state:
            torch.cuda.set_rng_state(state['active_cuda'], device)


def atomic_torch_save(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    # 同一文件系统内替换；新文件完整写入前，上一份断点始终可用。
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


@contextmanager
def _run_lock(directory):
    path = Path(directory) / '.training.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('此运行目录已有训练进程，不能同时续训。') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def _stop_signals():
    requested = {'signal': None}
    previous = {}

    def request_stop(number, frame):
        requested['signal'] = number

    if threading.current_thread() is threading.main_thread():
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            number = getattr(signal, name, None)
            if number is not None:
                handler = signal.getsignal(number)
                if name == 'SIGHUP' and handler == signal.SIG_IGN:
                    continue
                previous[number] = handler
                signal.signal(number, request_stop)
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _experiment_config(config):
    value = copy.deepcopy(config)
    for field in ('config_path', 'output', 'logging'):
        value.pop(field, None)
    for field in ('train', 'dev', 'test'):
        value.get('data', {}).pop(field, None)
    for field in ('cache_dir', 'expected_manifest_sha256'):
        value.get('teacher', {}).pop(field, None)
    value.get('etmier', {}).pop('source_dir', None)
    value.get('model', {}).pop('local_files_only', None)
    for field in ('save_every_steps', 'log_every_steps', 'threads'):
        value.get('training', {}).pop(field, None)
    return value


def validate_resume(state, metadata, documents, amp):
    if state.get('format_version') != 1:
        raise ValueError('断点缺少完整训练状态；旧版 best 权重不能用于断点续训。')
    saved = state['metadata']
    if _experiment_config(saved['config']) != _experiment_config(metadata['config']):
        raise ValueError('续训配置改变了模型或训练协议；请保留原 epochs/max_steps/种子/损失等设置。')
    for field in ('data_sha256', 'teacher_manifest_sha256', 'relation_vocab'):
        if saved.get(field) != metadata.get(field):
            raise ValueError('续训来源校验失败：' + field)
    old_upstream, new_upstream = copy.deepcopy(saved.get('upstream_provenance', {})), copy.deepcopy(metadata.get('upstream_provenance', {}))
    old_upstream.pop('source_dir', None)
    new_upstream.pop('source_dir', None)
    if old_upstream != new_upstream:
        raise ValueError('E1 源码或适配器身份与断点不一致。')
    if state['train_doc_ids'] != [doc['doc_id'] for doc in documents]:
        raise ValueError('续训文档集合或顺序发生变化。')
    if state['amp_enabled'] != amp:
        raise ValueError('断点的 AMP 启用状态不同；请在相同精度配置的设备上续训。')


def create_scheduler(optimizer, settings, total_steps):
    name = settings.get('scheduler', 'constant')
    warmup = int(settings.get('warmup_steps', 0))
    if name not in ('constant', 'linear') or warmup < 0 or warmup >= total_steps:
        raise ValueError('scheduler 仅支持 constant/linear，且 0 <= warmup_steps < 总步数。')

    def multiplier(step):
        if warmup and step < warmup:
            return float(step) / warmup
        if name == 'linear':
            return max(0., float(total_steps - step) / max(1, total_steps - warmup))
        return 1.

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _save_best(model, directory, metadata, epoch, step, score, clean, noisy, extra):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'model.safetensors.tmp'
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(temporary))
    os.replace(temporary, directory / 'model.safetensors')
    checkpoint = {key: metadata[key] for key in (
        'config', 'relation_vocab', 'data_sha256', 'teacher_manifest_sha256')}
    checkpoint.update(epoch=epoch + 1, optimizer_steps=step, selection_score=score)
    checkpoint.update(extra or {})
    write_json(directory / 'checkpoint.json', checkpoint)
    write_json(directory / 'dev_clean.json', clean)
    write_json(directory / 'dev_confusion20.json', noisy)


def run_training(model, optimizer, train_documents, config, run_dir, metadata,
                 training_step, evaluate_epoch, resume=None, checkpoint_extra=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(run_dir):
        return _run_training(model, optimizer, train_documents, config, run_dir, metadata,
                             training_step, evaluate_epoch, resume, checkpoint_extra)


def _run_training(model, optimizer, documents, config, run_dir, metadata,
                  training_step, evaluate_epoch, resume, checkpoint_extra):
    settings = config['training']
    accumulation = int(settings.get('accumulation_steps', 1))
    epochs = int(settings['epochs'])
    save_every = int(settings.get('save_every_steps', 100))
    log_every = int(settings.get('log_every_steps', 1))
    if min(accumulation, epochs, save_every, log_every) < 1:
        raise ValueError('累积步数、epochs、保存及日志间隔必须为正整数。')

    total_steps = math.ceil(len(documents) / accumulation) * epochs
    if settings.get('max_steps') is not None:
        total_steps = min(total_steps, int(settings['max_steps']))

    device = next(model.parameters()).device
    amp = bool(settings.get('amp', True) and device.type == 'cuda')
    scheduler = create_scheduler(optimizer, settings, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    state_path = run_dir / 'last' / 'training.pt'
    epoch, position, step, attempts = 0, 0, 0, 0
    best_score, elapsed = -math.inf, 0.
    history, epoch_losses, epoch_observations = [], [], []
    order = None
    saved_state = None
    if resume:
        _, state_path = resolve_resume_path(resume)
        if not state_path.exists():
            raise FileNotFoundError('缺少 last/training.pt；旧版仅保存模型权重的实验无法完整续训。')
        saved_state = torch.load(state_path, map_location='cpu', weights_only=True)
        validate_resume(saved_state, metadata, documents, amp)
        model.load_state_dict(saved_state['model'])
        # 先构建 scheduler，再恢复 optimizer，避免构造调度器覆盖已保存的学习率。
        optimizer.load_state_dict(saved_state['optimizer'])
        scheduler.load_state_dict(saved_state['scheduler'])
        scaler.load_state_dict(saved_state['scaler'])
        # 状态已载入，及时释放读取断点时的 CPU 权重副本。
        del saved_state['model'], saved_state['optimizer']
        epoch, position = saved_state['epoch'], saved_state['batch_position']
        step, attempts = saved_state['global_step'], saved_state['optimizer_attempts']
        order = saved_state['epoch_order']
        history, best_score = saved_state['history'], saved_state['best_score']
        epoch_losses, epoch_observations = saved_state['epoch_losses'], saved_state['epoch_observations']
        elapsed = saved_state['elapsed_seconds']
    elif (run_dir / 'run.json').exists():
        raise FileExistsError('运行目录已经存在；续训须显式使用 --resume。')

    logger = TrainingLogger(run_dir, config.get('logging', {}), resume_step=step if resume else None,
                            resume_state=saved_state.get('logging') if saved_state else None)
    start = time.perf_counter()
    metadata = json.loads(json.dumps(metadata))
    metadata.update(status='running', resume_count=int(saved_state['metadata'].get('resume_count', 0)) + 1 if resume else 0)
    write_json(run_dir / 'run.json', metadata)
    write_json(run_dir / 'relation_vocab.json', metadata['relation_vocab'])
    write_json(run_dir / 'history.json', history)
    if saved_state:
        restore_rng(saved_state['rng'], device)
        del saved_state

    def save_progress():
        logger.flush()
        state = {
            # 同步保存由 torch.save 逐个 storage 搬到 CPU，避免整个模型状态的额外常驻副本。
            'format_version': 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'amp_enabled': amp, 'rng': capture_rng(device),
            'logging': logger.state_dict(),
            'epoch': epoch, 'batch_position': position, 'epoch_order': order,
            'global_step': step, 'optimizer_attempts': attempts,
            'history': history, 'best_score': best_score,
            'epoch_losses': epoch_losses, 'epoch_observations': epoch_observations,
            'train_doc_ids': [doc['doc_id'] for doc in documents], 'metadata': metadata,
            'elapsed_seconds': elapsed + time.perf_counter() - start,
        }
        atomic_torch_save(state, state_path)

    optimizer.zero_grad(set_to_none=True)
    status = 'complete'
    try:
        if not resume:
            save_progress()

        with _stop_signals() as stopping:
            while epoch < epochs:
                if order is None:
                    if step >= total_steps:
                        break

                    order = list(range(len(documents)))
                    random.Random(int(settings['seed']) + epoch).shuffle(order)
                    if settings.get('max_steps') is not None:
                        order = order[:max(0, (int(settings['max_steps']) - step) * accumulation)]
                    position, epoch_losses, epoch_observations = 0, [], []

                model.train()
                while position < len(order):
                    group_end = min(len(order), position + accumulation)
                    group_size = group_end - position
                    group_losses = []
                    used_lrs = [group['lr'] for group in optimizer.param_groups]
                    for offset in range(position, group_end):
                        with torch.autocast(device_type=device.type, enabled=amp):
                            losses, observation = training_step(order[offset], epoch)
                            loss = losses['total'] / group_size
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('出现非有限损失；保留上一个完整更新边界的断点。')
                        scaler.scale(loss).backward()
                        group_losses.append({key: float(value.detach()) for key, value in losses.items()})
                        epoch_observations.append(observation)
                        del losses, loss

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get('max_grad_norm', 1.0),
                                                   error_if_nonfinite=not amp)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < previous_scale
                    optimizer.zero_grad(set_to_none=True)
                    if not skipped:
                        step += 1
                        scheduler.step()

                    attempts += 1
                    position = group_end
                    epoch_losses.extend(group_losses)
                    if attempts % log_every == 0 or position == len(order) or stopping['signal']:
                        values = {'train/loss_' + key: sum(row[key] for row in group_losses) / group_size
                                  for key in group_losses[0]}
                        values.update({'train/lr_encoder': used_lrs[0], 'train/lr_head': used_lrs[-1],
                                       'train/amp_scale': scaler.get_scale(), 'train/amp_skipped': float(skipped)})
                        logger.log(step, epoch + 1, values)

                    # 仅保存梯度累积完成的状态，恢复时不会漏算或重复累积半个 batch
                    if attempts % save_every == 0 or position == len(order) or stopping['signal']:
                        save_progress()

                    if stopping['signal']:
                        status = 'interrupted'
                        break

                if status == 'interrupted':
                    break

                clean, noisy = evaluate_epoch()
                score = (clean['metrics']['f1'] + noisy['metrics']['f1']) / 2
                record = {'epoch': epoch + 1, 'optimizer_steps': step, 'selection_score': score,
                          'dev_clean_f1': clean['metrics']['f1'], 'dev_confusion20_f1': noisy['metrics']['f1'],
                          'loss': {key: float(np.mean([row[key] for row in epoch_losses])) for key in epoch_losses[0]}}
                history.append(record)
                logger.log(step, epoch + 1, {
                    'dev/clean_f1': clean['metrics']['f1'], 'dev/confusion20_f1': noisy['metrics']['f1'],
                    'dev/clean_ign_f1': clean['metrics'].get('ign_f1', 0.),
                    'dev/confusion20_ign_f1': noisy['metrics'].get('ign_f1', 0.),
                    'dev/selection_score': score,
                })
                write_json(run_dir / 'history.json', history)
                write_json(run_dir / 'train_manifests' / ('epoch_%02d.json' % (epoch + 1)), epoch_observations)
                if score > best_score:
                    best_score = score
                    _save_best(model, run_dir / 'best', metadata, epoch, step, score, clean, noisy, checkpoint_extra)
                del clean, noisy
                epoch += 1
                position, order, epoch_losses, epoch_observations = 0, None, [], []
                save_progress()
                print(json.dumps(record, ensure_ascii=False), flush=True)

                if stopping['signal']:
                    status = 'interrupted'
                    break

    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error), recovery_checkpoint=str(state_path))
        write_json(run_dir / 'run.json', metadata)
        # 未完成的微批次、验证或写盘异常都从已提交断点重做，不保存不一致的瞬时状态
        raise

    finally:
        logger.close()

    metadata.update(status=status, optimizer_steps=step, completed_epochs=epoch,
                    best_dev_selection_score=best_score if math.isfinite(best_score) else None,
                    seconds=elapsed + time.perf_counter() - start,
                    recovery_checkpoint=str(state_path),
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    write_json(run_dir / 'run.json', metadata)

    return run_dir
