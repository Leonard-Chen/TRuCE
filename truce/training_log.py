#!/usr/bin/env python3
'''基于 CSV 和 TensorBoard 的标量日志记录模块'''

import csv
import io
import math
import os
import sys
import time
import types
from pathlib import Path, PureWindowsPath


FIELDS = ('wall_time', 'step', 'epoch', 'tag', 'value')


def prepare_tensorboard():
    # 本项目只需事件文件，使用 TensorBoard 自带文件接口，避免加载额外 TensorFlow 运行时
    sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))


def _tensorboard_directory(run_dir, value):
    from .runtime import _expand_path
    expanded = _expand_path(value)
    if os.name != 'nt' and PureWindowsPath(expanded).drive:
        raise ValueError('logging.tensorboard_dir 仍是 Windows 路径，请用 --set 指定云端路径或 tensorboard。')
    directory = Path(expanded)
    return directory if directory.is_absolute() else run_dir / directory


class TrainingLogger:
    def __init__(self, run_dir, settings=None, resume_step=None, resume_state=None):
        settings = settings or {}
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / 'metrics.csv'
        self.flush_seconds = float(settings.get('flush_seconds', 10))

        if not math.isfinite(self.flush_seconds) or self.flush_seconds <= 0:
            raise ValueError('logging.flush_seconds 必须大于零。')
        if resume_step is not None and int(resume_step) < 0:
            raise ValueError('resume_step 不能小于零。')
        if resume_state is not None and resume_step is None:
            raise ValueError('恢复已提交日志需要同时指定 resume_step。')

        self.writer = None
        self.stream = None
        writer_class = None
        directory = None

        if settings.get('tensorboard', True):
            directory = _tensorboard_directory(self.run_dir, settings.get('tensorboard_dir', 'tensorboard'))
            prepare_tensorboard()

            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    'TensorBoard 日志已启用，请安装 requirements.txt 中的 tensorboard；'
                    '或显式设置 logging.tensorboard=false。'
                ) from error

            writer_class = SummaryWriter

        self.run_dir.mkdir(parents=True, exist_ok=True)
        replay = {}
        if resume_state is not None:
            replay = self._restore_committed_csv(resume_state, int(resume_step))
        elif resume_step is not None and self.csv_path.exists():
            self._truncate_csv(int(resume_step))

        self.stream = self.csv_path.open('a', encoding='utf-8', newline='')
        self.csv_writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        if self.csv_path.stat().st_size == 0:
            self.csv_writer.writeheader()
            self.stream.flush()

        if writer_class is not None:
            try:
                purge_step = None if resume_step is None else int(resume_step) + 1
                if resume_state is not None:
                    purge_step = int(resume_step)
                self.writer = writer_class(
                    log_dir=str(directory),
                    purge_step=purge_step,
                    flush_secs=self.flush_seconds,
                )
                # 同一步也可能混入未提交验证记录：清除当前步后仅重放 CSV 中已提交的标量
                for tag, row in replay.items():
                    self.writer.add_scalar(tag, float(row['value']), global_step=int(resume_step),
                                           walltime=float(row['wall_time']))
                self.writer.flush()
            except Exception:
                self.stream.close()
                self.stream = None
                raise
        self.last_flush = time.monotonic()

    def _restore_committed_csv(self, state, step):
        position = state.get('csv_bytes') if isinstance(state, dict) else None

        if type(position) is not int or position <= 0:
            raise ValueError('断点中的 logging.csv_bytes 必须是有效的正整数字节位置。')
        if not self.csv_path.is_file():
            raise FileNotFoundError('断点恢复缺少已提交日志 metrics.csv，请连同运行目录一起迁移。')

        size = self.csv_path.stat().st_size
        if size < position:
            raise ValueError('metrics.csv 短于断点记录的已提交长度，不能恢复不完整日志。')
        with self.csv_path.open('rb') as stream:
            committed = stream.read(position)

        try:
            if not committed.endswith(b'\n'):
                raise ValueError('incomplete record')

            reader = csv.DictReader(io.StringIO(committed.decode('utf-8-sig'), newline=''), strict=True)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError('invalid header')

            replay = {}
            for row in reader:
                if (None in row or any(row.get(key) is None for key in FIELDS)
                        or not row['tag'] or not math.isfinite(float(row['value']))
                        or not math.isfinite(float(row['wall_time']))
                        or not 0 <= int(row['step']) <= step or int(row['epoch']) < 0):
                    raise ValueError('invalid committed record')
                if int(row['step']) == step:
                    replay[row['tag']] = row

        except (UnicodeError, csv.Error, TypeError, ValueError) as error:
            raise ValueError('metrics.csv 的已提交部分损坏，或断点字节位置未落在完整记录边界。') from error

        if size != position:
            temporary = self.csv_path.with_suffix('.csv.tmp')
            with temporary.open('wb') as stream:
                stream.write(committed)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.csv_path)

        return replay

    def _truncate_csv(self, step):
        with self.csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError('metrics.csv 表头与当前日志格式不一致。')
            rows = list(reader)
        kept = []

        for index, row in enumerate(rows):
            try:
                valid = (
                    None not in row and all(row.get(key) is not None for key in FIELDS)
                    and math.isfinite(float(row['value']))
                    and math.isfinite(float(row['wall_time']))
                )
                if not valid:
                    raise ValueError('incomplete row')
                row_step = int(row['step'])
                int(row['epoch'])

            except (TypeError, ValueError):
                # 崩溃时仅末行可能未写完；中间损坏不能静默略过
                if index == len(rows) - 1:
                    continue
                raise ValueError('metrics.csv 中存在损坏的非末尾记录。')

            if row_step <= step:
                kept.append(row)

        temporary = self.csv_path.with_suffix('.csv.tmp')

        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.csv_path)

    def log(self, step, epoch, values):
        if self.stream is None:
            raise RuntimeError('日志已关闭。')

        step, epoch = int(step), int(epoch)
        if step < 0 or epoch < 0:
            raise ValueError('日志 step 和 epoch 不能小于零。')

        scalars = {str(tag): float(value) for tag, value in values.items()}
        if any(not tag or not math.isfinite(value) for tag, value in scalars.items()):
            raise ValueError('日志 tag 不能为空，且指标必须为有限标量。')

        wall_time = time.time()
        for tag, value in scalars.items():
            self.csv_writer.writerow({
                'wall_time': wall_time, 'step': step, 'epoch': epoch,
                'tag': tag, 'value': value,
            })
            if self.writer is not None:
                self.writer.add_scalar(tag, value, global_step=step, walltime=wall_time)

        self.stream.flush()
        if time.monotonic() - self.last_flush >= self.flush_seconds:
            self.flush()

    def flush(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        if self.writer is not None:
            self.writer.flush()
        self.last_flush = time.monotonic()

    def state_dict(self):
        if self.stream is None:
            raise RuntimeError('日志已关闭，不能记录提交位置。')
        self.flush()
        return {'csv_bytes': self.csv_path.stat().st_size}

    def close(self):
        try:
            self.flush()
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
