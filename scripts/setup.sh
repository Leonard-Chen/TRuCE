#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"

usage() {
    cat <<'EOF'
用法：bash scripts/setup.sh [选项]
  --python PATH                 Python 3.9+，也可设置 TRUCE_PYTHON
  --venv-dir DIR                环境目录，默认 TRUCE_VENV_DIR 或 .venv
  --reuse-installed-packages    复用指定 Python 已安装的包
  --torch-index-url URL         PyTorch 源，默认 cu126，可改为官方 cpu 源
  -h, --help                    显示帮助
相对路径基于项目根目录。
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
torch_index_url=${TRUCE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
reuse_packages=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) need_value "$@"; python=$2; shift 2 ;;
        --venv-dir) need_value "$@"; venv_dir=$2; shift 2 ;;
        --torch-index-url) need_value "$@"; torch_index_url=$2; shift 2 ;;
        --reuse-installed-packages) reuse_packages=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf '未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

# 先检查系统库，防止在 CentOS 7 原生环境中下载完依赖才发现无法加载。
if [[ $(uname -s) == Linux ]]; then
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
    if [[ $libc =~ ^glibc[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        major=${BASH_REMATCH[1]}
        minor=${BASH_REMATCH[2]}
        if (( major < 2 || (major == 2 && minor < 28) )); then
            printf '当前 %s；PyTorch 2.8 需要 glibc 版本 >= 2.28。\n' "$libc" >&2
            exit 1
        fi
    else
        printf '无法确认 GNU glibc 版本，请在受支持的 Linux 或 PyTorch 容器中安装。\n' >&2
        exit 1
    fi
fi

if [[ -z $python ]]; then
    for candidate in python3.9 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
            python=$candidate
            break
        fi
    done
fi
if [[ -z $python ]] || ! command -v "$python" >/dev/null 2>&1; then
    printf '未找到 Python 3.9+，请通过 --python 指定解释器。\n' >&2
    exit 1
fi
"$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else "需要 Python 3.9 或更高版本。")'

[[ $venv_dir == /* ]] || venv_dir="$project_root/$venv_dir"
venv_python="$venv_dir/bin/python"
if [[ ! -x $venv_python ]]; then
    if [[ -e $venv_dir ]]; then
        printf '目录已存在但没有可用的 Linux Python：%s\n请指定新的 --venv-dir，不要复用 Windows 环境。\n' "$venv_dir" >&2
        exit 1
    fi
    venv_options=()
    if $reuse_packages; then
        venv_options+=(--system-site-packages)
    fi
    "$python" -m venv ${venv_options[@]+"${venv_options[@]}"} "$venv_dir"
fi

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
"$venv_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else "已有环境的 Python 版本低于 3.9。")'
"$venv_python" -m pip install --upgrade pip
# 固定已核验的框架版本，避免容器使用较新 Python 时解析到另一代 Transformers
"$venv_python" -m pip install 'torch==2.8.0' --index-url "$torch_index_url"
"$venv_python" -m pip install -r "$project_root/requirements.txt" 'torch==2.8.0' 'transformers==4.57.6'
"$venv_python" -m pip check
"$venv_python" -m truce doctor
printf '\n环境已就绪：%s\n' "$venv_python"
