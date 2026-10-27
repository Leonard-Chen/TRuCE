#!/usr/bin/env python3
'''准备冻结文本类型器及文档级 OOF 缓存'''

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truce.runtime import load_config
from truce.teachers import prepare_teacher_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--output')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke', action='store_true')

    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or config.get('teacher', {}).get('cache_dir', 'artifacts/teachers')
    result = prepare_teacher_artifacts(config, output, smoke=args.smoke, device=args.device)
    print('类型器准备完成：{}；{}'.format(output, result['status']))


if __name__ == '__main__':
    main()
