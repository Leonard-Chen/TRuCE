#!/usr/bin/env python3
'''下载固定版本的 Re-DocRED 官方数据集'''

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from truce.data import build_relation_vocab, normalize_document


COMMIT = 'ccfb54f5ddf5836027c87badda10f6dfc56efaac'
BASE_URL = 'https://raw.githubusercontent.com/tonytan48/Re-DocRED/' + COMMIT
FILES = {
    'train': ('train_revised.json', 'c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2', 3053),
    'dev': ('dev_revised.json', '051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f', 500),
    'test': ('test_revised.json', 'ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c', 500),
}


def download(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': 'https://github.com/tonytan48/Re-DocRED', 'commit': COMMIT,
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'files': {},
                'scope': 'Official annotated train/dev/test only; no distant supervision'}
    relation_vocab = None

    for split, (name, expected_sha, expected_count) in FILES.items():
        path = output_dir / name
        url = BASE_URL + '/data/' + name
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'TRuCE-research-scaffold'})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()

        digest = hashlib.sha256(content).hexdigest()
        if digest != expected_sha:
            raise RuntimeError('{} SHA256 不匹配；保留已有文件且停止，不自动覆盖'.format(path))

        documents = json.loads(content)
        if len(documents) != expected_count:
            raise RuntimeError('{} 文档数量不匹配'.format(split))
        if len({document['title'] for document in documents}) != len(documents):
            raise RuntimeError('官方 title 不唯一')

        normalized = [normalize_document(document, document['title']) for document in documents]

        if not path.exists():
            temporary = path.with_suffix('.download')
            temporary.write_bytes(content)
            temporary.replace(path)
        if split == 'train':
            relation_vocab = build_relation_vocab(documents)

        manifest['files'][split] = {'file': name, 'url': url, 'sha256': digest,
                                    'bytes': len(content), 'documents': len(documents),
                                    'entities': sum(len(doc['vertexSet']) for doc in normalized),
                                    'inconsistent_type_clusters': sum(not valid for doc in normalized for valid in doc['class_consistent']),
                                    'relations': sum(len(doc['labels']) for doc in normalized),
                                    'relations_without_evidence': sum(not label.get('evidence') for doc in normalized for label in doc['labels'])}

        print('{}: {} documents, SHA256 verified'.format(split, len(documents)), flush=True)

    (output_dir / 'relation_vocab.json').write_text(json.dumps(relation_vocab, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved: {}'.format(output_dir.resolve()), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw')
    args = parser.parse_args()
    download(args.output_dir)
