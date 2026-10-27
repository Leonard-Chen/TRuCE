#!/usr/bin/env python3
'''基于 CSV/TensorBoard 日志绘制图表'''

import argparse
import csv
import json
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _series(rows):
    latest = {}
    for row in rows:
        step, value = int(row['step']), float(row['value'])
        if not math.isfinite(value):
            continue
        tag = row['tag']
        key = (tag, step)
        # 续训追加的同一步记录覆盖旧值，不依赖迁移前后系统时钟完全一致
        latest[key] = value

    result = {}
    for (tag, step), value in sorted(latest.items()):
        result.setdefault(tag, []).append((step, value))
    return result


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'step', 'tag', 'value'} <= set(reader.fieldnames or ()):
            raise ValueError('CSV 须为包含 step、tag、value 的长表日志。')
        rows = list(reader)
    return _series(rows)


def read_tensorboard(path):
    from truce.training_log import prepare_tensorboard
    prepare_tensorboard()

    try:
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.util import tensor_util

    except ImportError as error:
        raise RuntimeError('读取事件日志需要安装 requirements.txt 中的 tensorboard。') from error

    accumulator = event_accumulator.EventAccumulator(
        str(path), size_guidance={'scalars': 0, 'tensors': 0}, purge_orphaned_data=True,
    ).Reload()

    rows = []
    for tag in accumulator.Tags().get('scalars', []):
        rows.extend({
            'tag': tag, 'step': event.step, 'value': event.value, 'wall_time': event.wall_time,
        } for event in accumulator.Scalars(tag))

    for tag in accumulator.Tags().get('tensors', []):
        for event in accumulator.Tensors(tag):
            value = tensor_util.make_ndarray(event.tensor_proto)
            if value.size == 1 and value.dtype.kind in 'biuf':
                rows.append({
                    'tag': tag, 'step': event.step, 'value': float(value.reshape(-1)[0]),
                    'wall_time': event.wall_time,
                })
    return _series(rows)


def plot_metrics(series, output_dir, tags=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    selected = set(tags) if tags else set(series)
    missing = selected - set(series)
    if missing:
        raise ValueError('日志中没有这些指标：' + ', '.join(sorted(missing)))
    if not selected:
        raise ValueError('未读取到可绘制的标量日志。')

    groups = {'loss': [], 'learning_rate': [], 'dev': [], 'other': []}
    for tag in sorted(selected):
        if tag.startswith('dev/'):
            group = 'dev'
        elif '/lr' in tag or 'learning_rate' in tag:
            group = 'learning_rate'
        elif 'loss' in tag:
            group = 'loss'
        else:
            group = 'other'
        groups[group].append(tag)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    labels = {'loss': 'Loss', 'learning_rate': 'Learning rate', 'dev': 'Metric (0-1)', 'other': 'Value'}

    for group, group_tags in groups.items():
        if not group_tags:
            continue

        figure, axes = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        for tag in group_tags:
            points = series[tag]
            axes.plot([point[0] for point in points], [point[1] for point in points],
                      label=tag, linewidth=1.5, marker='o' if len(points) <= 20 else None,
                      markersize=3)

        axes.set_xlabel('Optimizer step')
        axes.set_ylabel(labels[group])
        axes.grid(alpha=0.25)
        axes.legend(fontsize=8)

        for extension in ('png', 'pdf'):
            destination = directory / (group + '.' + extension)
            figure.savefig(destination, dpi=180)
            outputs.append(str(destination.resolve()))

        plt.close(figure)

    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description='独立绘制训练损失、学习率和开发集指标')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', help='运行目录内的 metrics.csv')
    source.add_argument('--tensorboard', help='单个运行的 TensorBoard 目录或事件文件')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tags', nargs='+', help='可选：只画指定 tag，例如 train/loss_total dev/clean_f1')

    args = parser.parse_args(argv)
    series = read_csv(args.csv) if args.csv else read_tensorboard(args.tensorboard)
    outputs = plot_metrics(series, args.output_dir, args.tags)
    print(json.dumps({'figures': outputs}, ensure_ascii=False, indent=2))
    return outputs


if __name__ == '__main__':
    main()
