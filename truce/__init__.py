#!/usr/bin/env python3

import os

# 仅启用本项目使用的 PyTorch 后端，避免载入无关的可选 TensorFlow/Flax 包
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

__version__ = '0.1.0'
