#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"

usage() {
    cat <<'EOF'
用法：bash scripts/run_smoke.sh [选项]
  --output DIR        输出目录，默认 runs/smoke；必须为空或不存在
  --device DEVICE     auto、cpu 或 cuda，默认 auto
  --python PATH       已安装依赖的 Python，默认使用项目 Linux 环境
  --venv-dir DIR      环境目录，默认 TRUCE_VENV_DIR 或 .venv
  --include-etmier    同时验证 E1，需先准备其外部源码
  -h, --help          显示帮助
相对路径基于项目根目录。本脚本仅验证流程，不运行正式 BERT 实验。
EOF
}

need_value() {
    if [[ $# -lt 2 || -z $2 || $2 == --* ]]; then
        printf '参数 %s 缺少值。\n' "$1" >&2
        exit 2
    fi
}

python=${TRUCE_PYTHON:-}
venv_dir=${TRUCE_VENV_DIR:-.venv}
output=runs/smoke
device=auto
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) need_value "$@"; output=$2; shift 2 ;;
        --device) need_value "$@"; device=$2; shift 2 ;;
        --python) need_value "$@"; python=$2; shift 2 ;;
        --venv-dir) need_value "$@"; venv_dir=$2; shift 2 ;;
        --include-etmier) extra+=(--include-etmier); shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf '未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z $python ]]; then
    [[ $venv_dir == /* ]] || venv_dir="$project_root/$venv_dir"
    python="$venv_dir/bin/python"
fi
if ! command -v "$python" >/dev/null 2>&1; then
    printf '没有可用的 Python：%s\n请先运行 setup.sh，或通过 --python 指定已配置的环境。\n' "$python" >&2
    exit 1
fi

# 烟测使用独立的微型数据与教师缓存，不继承正式实验的路径和编码器覆盖。
unset DOCRE_DATA_DIR DOCRE_TEACHER_DIR DOCRE_OUTPUT_DIR DOCRE_ENCODER_NAME
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
exec "$python" -u -m truce smoke --output "$output" --device "$device" ${extra[@]+"${extra[@]}"}
