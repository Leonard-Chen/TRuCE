#!/usr/bin/env python3
'''命令行启动脚本'''

import argparse
import copy
import json
from pathlib import Path

from .runtime import (ROOT, choose_device, environment_info, load_config, load_split,
                      read_json, resolve_path, sha256, write_json, deep_update, load_resume_config)

CONDITIONS = ['clean', 'missing', 'uniform', 'confusion', 'local']


def parser():
    result = argparse.ArgumentParser(description='TRuCE 训练、评估与可复现类型干预')
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='查看 Python/PyTorch 环境及自动选择的设备')

    for name in ('train', 'audit', 'intervene', 'matrix'):
        item = commands.add_parser(name)
        item.add_argument('--config', default=None if name == 'train' else str(ROOT / 'configs' / 'B0.yaml'))
        item.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
        item.add_argument('--output')
        if name in ('train', 'matrix'):
            item.add_argument('--device', default='auto')
        if name == 'train':
            item.add_argument('--resume', help='从运行目录、last 目录或 last/training.pt 恢复完整训练状态')
        if name == 'matrix':
            item.add_argument('--variants', nargs='+', default=['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'E1'])
            item.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
            item.add_argument('--execute', action='store_true', help='默认只生成计划；此参数才逐项执行')
        if name == 'intervene':
            item.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
            item.add_argument('--condition', choices=CONDITIONS, default='confusion')
            item.add_argument('--rate', type=float, default=0.2)
            item.add_argument('--noise-seed', type=int, default=101)

    for name in ('evaluate', 'sweep', 'diagnose', 'evidence'):
        item = commands.add_parser(name)
        item.add_argument('--checkpoint', required=True)
        item.add_argument('--split', default='test' if name in ('evaluate', 'sweep') else 'dev', choices=['train', 'dev', 'test'])
        item.add_argument('--device', default='auto')
        item.add_argument('--output', required=True)
        item.add_argument('--limit', type=int)
        if name == 'evaluate':
            item.add_argument('--condition', choices=CONDITIONS, default='clean')
            item.add_argument('--rate', type=float, default=0.0)
            item.add_argument('--noise-seed', type=int, default=101)
        if name == 'evidence':
            item.add_argument('--source', choices=['predicted', 'reference'], default='predicted')

    item = commands.add_parser('bootstrap')
    item.add_argument('--config', default=str(ROOT / 'configs' / 'B0.yaml'))
    item.add_argument('--a', required=True)
    item.add_argument('--b', required=True)
    item.add_argument('--output', required=True)
    item.add_argument('--iterations', type=int, default=1000)
    item = commands.add_parser('smoke', help='生成微型数据，验证全部配置的训练、加载、干预及评估')
    item.add_argument('--output', default='runs/smoke')
    item.add_argument('--device', default='auto')
    item.add_argument('--include-etmier', action='store_true', help='取得固定源码后，同时验证独立 E1')

    return result


def audit(config):
    from .data import build_relation_vocab
    vocab = build_relation_vocab(load_split(config['data']['train']))
    report = {'relation_classes': len(vocab), 'splits': {}}
    for split in ('train', 'dev', 'test'):
        docs = load_split(config['data'][split])
        report['splits'][split] = {
            'documents': len(docs), 'entities': sum(len(d['vertexSet']) for d in docs),
            'inconsistent_entities': sum(not x for d in docs for x in d['class_consistent']),
            'relations': sum(len(d['labels']) for d in docs),
            'empty_evidence_relations': sum(not x.get('evidence') for d in docs for x in d['labels']),
            'source_sha256': sha256(config['data'][split]),
            'max_raw_words': max(sum(len(s) for s in d['sents']) for d in docs),
            'max_entities': max(len(d['vertexSet']) for d in docs),
        }
    return report


def matrix(config, variants, seeds, output, execute, device):
    from .engine import train
    allowed = {'B' + str(i) for i in range(9)} | {'E1'}
    if not set(variants) <= allowed:
        raise ValueError('支持的实验配置为 B0–B8、E1。')

    directory = resolve_path(output or 'runs/matrix')
    jobs = []
    for variant in variants:
        for seed in seeds:
            current = copy.deepcopy(config)
            current['model']['variant'] = variant
            current['training']['seed'] = seed

            if variant == 'E1':
                filename = 'E1_smoke.yaml' if current['model'].get('encoder_backend') == 'tiny' else 'E1.yaml'
                defaults = load_config(ROOT / 'configs' / filename)
                current['etmier'] = deep_update(defaults['etmier'], current.get('etmier', {}))
                # 原注意力归一化含极小常数；E1 默认保留全精度，避免半精度下溢。
                current['training']['amp'] = False

            destination = directory / (variant + '_seed' + str(seed))
            job = {'variant': variant, 'seed': seed, 'output': str(destination),
                   'status': 'planned', 'config': current}
            jobs.append(job)

    manifest = directory / 'matrix.json'
    if manifest.exists():
        raise FileExistsError('矩阵目录已有计划，请换一个输出目录。')

    write_json(manifest, jobs)
    if execute:
        for job in jobs:
            run_directory = train(job['config'], job['output'], device)
            job['status'] = read_json(Path(run_directory) / 'run.json')['status']
            write_json(manifest, jobs)
            if job['status'] != 'complete':
                break

    return {'runs': len(jobs), 'executed': execute, 'manifest': str(manifest)}


def main(argv=None):
    args = parser().parse_args(argv)

    if args.command == 'doctor':
        result = environment_info()
        result['selected_device'] = str(choose_device())

    elif args.command == 'smoke':
        from .smoke import run_smoke
        result = run_smoke(args.output, args.device, args.include_etmier)

    elif args.command in ('train', 'audit', 'intervene', 'matrix'):
        if args.command == 'train' and args.resume:
            config = load_resume_config(args.resume, args.config, args.set)
        else:
            config = load_config(args.config or ROOT / 'configs' / 'B0.yaml', args.set)
        if args.command == 'train':
            from .engine import train
            result = {'run_directory': str(train(config, args.output, args.device, resume=args.resume))}
        elif args.command == 'audit':
            result = audit(config)
            if args.output:
                write_json(args.output, result)
        elif args.command == 'matrix':
            result = matrix(config, args.variants, args.seeds, args.output, args.execute, args.device)
        else:
            from .engine import TeacherStore, observation_for
            docs = load_split(config['data'][args.split])
            store = TeacherStore(config)
            result = {'split': args.split, 'source_sha256': sha256(config['data'][args.split]),
                      'observations': [observation_for(d, args.condition, args.rate, args.noise_seed, store, args.split)
                                       for d in docs]}
            if not args.output:
                raise ValueError('intervene 需要 --output 保存可复查清单。')
            write_json(args.output, result)
            result = {'manifest': args.output, 'documents': len(docs), 'sha256': sha256(args.output)}

    elif args.command == 'evaluate':
        from .engine import evaluate
        result = evaluate(args.checkpoint, args.split, args.condition, args.rate, args.noise_seed,
                          args.output, args.device, args.limit)['metrics']

    elif args.command == 'sweep':
        from .experiments import sweep
        result = sweep(args.checkpoint, args.split, args.output, args.device, args.limit)

    elif args.command == 'diagnose':
        from .experiments import diagnose
        result = diagnose(args.checkpoint, args.split, args.output, args.device, args.limit or 10)

    elif args.command == 'evidence':
        from .experiments import evidence_sensitivity
        result = evidence_sensitivity(args.checkpoint, args.split, args.output, args.device,
                                      args.limit or 100, args.source)

    else:
        from .metrics import build_train_facts, paired_bootstrap
        config = load_config(args.config)
        a, b = read_json(args.a), read_json(args.b)
        for field in ('split', 'source_sha256', 'doc_ids', 'condition', 'rate', 'noise_seed'):
            if a.get(field) != b.get(field):
                raise ValueError('bootstrap 的两份结果不匹配：' + field)
        if not a.get('source_sha256') or sha256(config['data'][a['split']]) != a['source_sha256']:
            raise ValueError('bootstrap 当前数据与原预测版本不一致，不能替换参考标签重算区间。')
        docs = load_split(config['data'][a['split']])
        if a.get('doc_ids') is not None:
            selected = set(a['doc_ids'])
            docs = [doc for doc in docs if doc['doc_id'] in selected]
        facts = build_train_facts(load_split(config['data']['train']))
        result = paired_bootstrap(docs, a['predictions'], b['predictions'], facts,
                                  iterations=args.iterations, seed=17)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
