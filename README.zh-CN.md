# TRuCE：类型与文本证据冲突的文档级关系抽取

[English](./README.md) | **简体中文**

## 环境要求

- Python 3.9 及以上
- GPU：NVIDIA GeForce RTX 3090 (24GB)
- CPU：14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- 依赖：
  - [PyTorch 2.8](https://pypi.org/project/torch/2.8.0/)
  - [Transformers 4.57.6](https://pypi.org/project/transformers/4.57.6/)
  - [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)
  - [Re-DocRED](https://github.com/tonytan48/Re-DocRED)

以下命令均在项目根目录执行。Windows PowerShell 下：

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

CentOS Linux 7 下：

```bash
bash scripts/setup.sh --python python3
source .venv/bin/activate
```

## 数据集

```
TRuCE
 |-- data
 |    |-- raw                     # 正式 Re-DocRED 数据集
 |    |    |-- train_revised.json
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |    |-- relation_vocab.json
 |    |    |-- source_manifest.json
 |    |-- toy                     # smoke 测试用数据集
 |    |    |-- train.json
 |    |    |-- dev.json
 |    |    |-- test.json
```

其中，`data/toy/` 供 smoke 测试用，`data/raw/` 供正式实验用。

## 输出目录结构

以下目录对应本文各命令的 `--output` 示例，执行相应命令后才会生成。训练未指定 `--output` 时，会在配置的输出目录下自动创建 `<变体>_seed<种子>_<时间戳>` 目录。

```text
runs/
 |-- first_smoke/
 |    |-- teachers/
 |    |-- B0/                     # B0 至 B8，每个配置一个目录
 |    |-- ...
 |    |-- B8/
 |    |-- smoke_report.json
 |    |-- evidence_sensitivity.json
 |-- all_smoke/                   # 结构同上，另含 E1/
 |-- B0_seed17/
 |    |-- .training.lock
 |    |-- run.json
 |    |-- relation_vocab.json
 |    |-- history.json
 |    |-- train_manifests/
 |    |    |-- epoch_01.json      # 每轮验证后保存该轮训练记录
 |    |    |-- ...
 |    |-- best/
 |    |    |-- model.safetensors
 |    |    |-- checkpoint.json
 |    |    |-- dev_clean.json
 |    |    |-- dev_confusion20.json
 |    |-- last/training.pt
 |    |-- metrics.csv
 |    |-- tensorboard/
 |         |-- events.out.tfevents.*

artifacts/
 |-- teachers/                   # prepare_teachers.py 生成
 |    |-- manifest.json
 |    |-- train.json
 |    |-- dev.json
 |    |-- test.json
 |    |-- confusion.json
 |    |-- *.safetensors          # 局部/全文类型头权重
 |    |-- *.npz                  # 冻结特征与 OOF logits
```

smoke 中每个配置的目录包含上述训练文件，另有 `test_<condition>.json`、`test_<condition>.predictions.json` 和 `diagnostics.json`。smoke 的教师缓存保存在该次 smoke 目录内部，与 `artifacts/teachers/` 中的正式教师缓存分开。

`best/` 在成功完成开发集评估后生成；`last/training.pt` 保存用于续训的完整状态。TensorBoard 默认开启，也可以关闭或改用其他目录。PNG/PDF 图片仅在单独运行 `scripts/plot_metrics.py` 后生成，保存位置由其 `--output-dir` 指定，训练本身不自动生成图片。

## 运行步骤

### 1. 环境自检

```bash
python -m truce doctor
python -m truce smoke --output runs/first_smoke
```

同时验证 E1：

```bash
python -m truce smoke --include-etmier --output runs/all_smoke
```

>注意，E1 需要固定版本的 ET-MIER 源码：
>
>```bash
>git clone --config core.autocrlf=false https://github.com/xin-fight/ET-MIER.git external/ET-MIER
>git -C external/ET-MIER checkout --detach c0650a930b065709e1c18d93400486b4be03832c
>```

继续已有训练：

```bash
python -m truce train --resume "runs/B8_seed17"
```

### 2. 准备

```bash
python scripts/download_data.py --output-dir data/raw
python -m truce audit --config configs/B0.yaml --output artifacts/data_audit.json
python scripts/prepare_teachers.py --config configs/B0.yaml --output artifacts/teachers
```

### 3. 训练

```bash
python -m truce train --config configs/B0.yaml --output runs/B0_seed17
python -m truce train --config configs/B1.yaml --output runs/B1_seed17
python -m truce train --config configs/B8.yaml --output runs/B8_seed17
```

小样本接口验证：

```bash
python -m truce train --config configs/B1.yaml --output runs/B1_pilot --set training.limit_train=32 --set training.limit_dev=16 --set training.max_steps=10 --set training.epochs=2
```

实验配置：

| 配置 | 处理                      |
| ---- |-------------------------|
| B0   | 不使用外部类型，保留文本与图底座        |
| B1   | 直接使用观测类型                |
| B2   | B1 + type-dropout=0.2   |
| B3   | B1 + 类型残差固定乘 lambda=0.5 |
| B4   | B1 + 全部替换为 q_doc        |
| B5   | B1 + 类型传播完后再做门控         |
| B6   | B1 + C（软纠正）             |
| B7   | B1 + G（传播前抑制）           |
| ★B8  | B1 + C + G（软纠正+传播前抑制）   |
| E1   | ET-MIER 的受控单模型适配        |

实验计划：

```bash
python -m truce matrix --config configs/B0.yaml --output runs/main_plan
```

执行（使用与计划预览不同的新目录）：

```bash
python -m truce matrix --config configs/B0.yaml --output runs/main_runs --execute
```

小范围执行（可选）：

```bash
python -m truce matrix --config configs/B0.yaml --output runs/pilot_runs --variants B0 B1 B3 B8 --seeds 17 --execute
```

不带 `--execute` 只会生成计划；已包含 `matrix.json` 的目录不能再次用于生成或执行矩阵。批量训练不会自动运行 `sweep`，模型保存在 `runs/main_runs/<配方>_seed<种子>/` 下面；下文的独立训练目录与其不同，评估时需使用实际路径。例如批量训练完成后：

```bash
python -m truce sweep --checkpoint runs/main_runs/B8_seed17 --split dev --output runs/main_runs/B8_seed17/dev_sweep
```

批量任务中断后，使用 `train --resume "runs/main_runs/B8_seed17"` 恢复实际中断的那次训练；当前 `matrix` 不负责自动恢复或接着执行剩余任务。

E1：

```bash
python -m truce train --config configs/E1.yaml --output runs/E1_seed17
python -m truce sweep --checkpoint runs/E1_seed17 --output runs/E1_seed17/noise_sweep
python -m truce diagnose --checkpoint runs/E1_seed17 --output runs/E1_seed17/diagnostics.json
```

### 4. 类型干预与评估

| 条件 | 名称      | 说明                    |
| ---- | --------- |-----------------------|
| C0   | clean     | 原始参考类型，不作人为改动         |
| C1   | missing   | 一部分实体的类型缺失            |
| C2   | uniform   | 一部分实体的类型被随机改错         |
| C3   | confusion | 按类型器常见的混淆规律改错         |
| C4   | local     | 使用局部类型器实际预测的类型，产生自然错误 |

```bash
python -m truce intervene --config configs/B8.yaml --split test --condition confusion --rate 0.2 --noise-seed 101 --output artifacts/test_C3_20_seed101.json
python -m truce evaluate --checkpoint runs/B1_seed17 --condition confusion --rate 0.2 --noise-seed 101 --output runs/B1_seed17/test_C3_20.json
python -m truce evaluate --checkpoint runs/B8_seed17 --condition confusion --rate 0.2 --noise-seed 101 --output runs/B8_seed17/test_C3_20.json
python -m truce sweep --checkpoint runs/B8_seed17 --output runs/B8_seed17/noise_sweep
python -m truce diagnose --checkpoint runs/B8_seed17 --output runs/B8_seed17/diagnostics.json
```

配对 bootstrap：

```bash
python -m truce bootstrap --config configs/B8.yaml --a runs/B1_seed17/test_C3_20.json --b runs/B8_seed17/test_C3_20.json --output runs/bootstrap_B8_minus_B1.json
```

### 5. C5 证据敏感性

屏蔽部分文本证据，检查模型是否依赖它们。

```bash
python -m truce evidence --checkpoint runs/B8_seed17 --split dev --source predicted --limit 100 --output runs/B8_seed17/evidence_predicted.json
python -m truce evidence --checkpoint runs/B8_seed17 --split dev --source reference --limit 100 --output runs/B8_seed17/evidence_reference.json
```

## 小结

- `smoke` 使用随机初始化小编码器，仅用于跑通流程，不能作为论文结果。
- 不要将 `artifacts/toy_teachers` 或 smoke 缓存用于正式数据。
- E1 中外部类型只进入辅助训练损失，推理时类型由文本预测；主要机制证据来自 B1/B6/B7/B8。
- E1 加载时会核验适配器的原始文件哈希；新增 shebang 等源码字节变化也会影响旧断点校验。恢复旧 E1 实验时，应使用与该断点匹配的代码版本，不修改断点里的来源哈希。
- 当前代码仓库结构仍有优化空间，后续会调整目录层级和脚本模块化。
