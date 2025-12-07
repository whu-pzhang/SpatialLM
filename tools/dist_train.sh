#!/usr/bin/env bash

# 默认值
MASTER_ADDR="127.0.0.1"
MASTER_PORT=""
GPUS=""
CONFIG=""

# 打印帮助信息
usage() {
    echo "Usage: $0 [options] <config_path>"
    echo ""
    echo "Options:"
    echo "  -g, --gpus <ids>    Specify GPU IDs (e.g., '0,1' or '4,5,6,7'). Default: all available."
    echo "  -p, --port <port>   Specify master port. Default: auto-detect free port."
    echo "  -h, --help          Show this help message."
    echo ""
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$CONFIG" ]; then
                CONFIG="$1"
                shift
            else
                echo "Error: Unknown argument '$1'"
                usage
            fi
            ;;
    esac
done

# 检查必填参数
if [ -z "$CONFIG" ]; then
    echo "Error: Configuration file path is required."
    usage
fi

# 自动寻找空闲端口
get_free_port() {
  local start=29500
  local end=29600
  for ((port=start; port<=end; port++)); do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -tuln 2>/dev/null | grep -q ":$port "; then
        echo $port
        return 0
      fi
    elif command -v netstat >/dev/null 2>&1; then
      if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo $port
        return 0
      fi
    else
      # 如果都没有，就随机返回一个
      echo $((start + RANDOM % (end - start)))
      return 0
    fi
  done
  echo "No free port found in range ${start}-${end}." >&2
  exit 1
}

if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=$(get_free_port)
fi

# 设置 GPU 相关变量
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    # 计算 GPU 数量 (逗号分隔的字段数)
    NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -v '^$' | wc -l)
else
    # 如果未指定 GPU，尝试使用 nvidia-smi 获取数量，或者默认为 1
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC=$(nvidia-smi -L | wc -l)
    else
        NPROC=1
        echo "[WARN] nvidia-smi not found, defaulting to NPROC_PER_NODE=1"
    fi
fi

echo "[INFO] Config: $CONFIG"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES:-All} (Count: $NPROC)"
echo "[INFO] Master Addr: $MASTER_ADDR"
echo "[INFO] Master Port: $MASTER_PORT"

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=$NPROC
export HF_ENDPOINT=https://hf-mirror.com
export FORCE_TORCHRUN=1

# 运行训练
PYTHONPATH=$(pwd):$PYTHONPATH \
    python train.py "$CONFIG"
