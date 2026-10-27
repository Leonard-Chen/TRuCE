#!/usr/bin/env python3
import copy
import time
from pathlib import Path

from .runtime import ROOT, load_config, resolve_path, write_json


def create_toy_dataset(directory):
    directory = Path(directory)
    for split, count in (('train', 12), ('dev', 3), ('test', 3)):
        docs = []
        for index in range(count):
            name = 'Alice' if index % 2 == 0 else 'Bob'
            sentences = [
                [name, 'joined', 'Acme', 'in', 'Paris', 'during', '2020', '.'],
                ['Acme', 'reported', '42', 'employees', 'for', 'ProjectX', '.'],
                ['The', 'company', 'develops', 'software', 'and', 'works', 'with', 'local', 'organizations', '.'],
                [name, 'continues', 'to', 'lead', 'ProjectX', 'from', 'Paris', '.'],
            ]
            specs = [(name, 'PER', [(0, 0, 1), (3, 0, 1)]),
                     ('Acme', 'ORG', [(0, 2, 3), (1, 0, 1)]),
                     ('Paris', 'LOC', [(0, 4, 5), (3, 6, 7)]),
                     ('42', 'NUM', [(1, 2, 3)]),
                     ('2020', 'TIME', [(0, 6, 7)]),
                     ('ProjectX', 'MISC', [(1, 5, 6), (3, 4, 5)])]
            entities = [[{'name': entity_name, 'type': typ, 'sent_id': sid, 'pos': [begin, end]}
                         for sid, begin, end in mentions] for entity_name, typ, mentions in specs]
            if index == 0:
                entities[1][1]['type'] = 'MISC'
            docs.append({'title': 'toy_%s_%02d' % (split, index), 'sents': sentences, 'vertexSet': entities,
                         'labels': [{'h': 0, 't': 1, 'r': 'P108', 'evidence': [0]},
                                    {'h': 1, 't': 2, 'r': 'P159', 'evidence': []}]})

        write_json(directory / (split + '.json'), docs)


def run_smoke(output='runs/smoke', device='auto', include_etmier=False):
    from .engine import train, evaluate
    from .experiments import diagnose, evidence_sensitivity
    from .teachers import prepare_teacher_artifacts
    start = time.perf_counter()
    directory = resolve_path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('小样本运行目录非空，请指定新的 --output。')

    directory.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / 'configs' / 'smoke.yaml')
    create_toy_dataset(ROOT / 'data' / 'toy')
    config['teacher']['cache_dir'] = str(directory / 'teachers')
    prepare_teacher_artifacts(config, directory / 'teachers', smoke=True, device=device)
    results = []
    variants = ['B' + str(i) for i in range(9)] + (['E1'] if include_etmier else [])
    for variant in variants:
        current = copy.deepcopy(config)
        current['model']['variant'] = variant
        if variant == 'E1':
            current['etmier'] = load_config(ROOT / 'configs' / 'E1_smoke.yaml')['etmier']

        run = train(current, directory / variant, device)
        for condition, rate in (('clean', 0.0), ('missing', 0.2), ('uniform', 0.2), ('confusion', 0.2), ('local', 0.0)):
            evaluated = evaluate(run, condition=condition, rate=rate, device=device,
                                 output=run / ('test_' + condition + '.json'))
            results.append({'variant': variant, 'condition': condition, 'f1': evaluated['metrics']['f1']})

        diagnose(run, 'dev', run / 'diagnostics.json', device, limit=2)

    evidence = evidence_sensitivity(directory / 'B8', 'dev', directory / 'evidence_sensitivity.json',
                                    device, limit=1, source='reference')

    result = {'status': 'passed', 'scope': '仅随机 tiny 编码器的流程验证，不代表 BERT 性能或论文实验',
              'variants': len(variants), 'evaluation_conditions': len(results), 'results': results,
              'evidence_examples': evidence['eligible_examples'], 'seconds': time.perf_counter() - start,
              'output': str(directory)}
    write_json(directory / 'smoke_report.json', result)
    return result
