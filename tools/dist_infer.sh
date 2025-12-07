#!/usr/bin/env bash

# 默认值
MASTER_ADDR="127.0.0.1"
MASTER_PORT=""
GPUS=""
POINT_CLOUD=""
OUTPUT=""
MODEL_PATH="manycore-research/SpatialLM-Llama-1B"
EXTRA_ARGS=""

# 打印帮助信息
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -g, --gpus <ids>            Specify GPU IDs (e.g., '0,1' or '4,5,6,7'). Default: all available."
    echo "  -p, --point_cloud <path>    Path to input point cloud file or directory."
    echo "  -o, --output <path>         Path to output directory or file."
    echo "  -m, --model_path <path>     Path to model checkpoint. Default: ${MODEL_PATH}"
    echo "  --port <port>               Specify master port. Default: auto-detect free port."
    echo "  -h, --help                  Show this help message."
    echo "  [other args]                Other arguments passed to inference_custom.py (e.g., --detect_type, --category)"
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
        -p|--point_cloud)
            POINT_CLOUD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            # 收集其他参数
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 检查必填参数
if [ -z "$POINT_CLOUD" ] || [ -z "$OUTPUT" ]; then
    echo "Error: Point cloud path (-p/--point_cloud) and output path (-o/--output) are required."
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

echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES:-All} (Count: $NPROC)"
echo "[INFO] Point Cloud: $POINT_CLOUD"
echo "[INFO] Output: $OUTPUT"
echo "[INFO] Model Path: $MODEL_PATH"
echo "[INFO] Master Addr: $MASTER_ADDR"
echo "[INFO] Master Port: $MASTER_PORT"
echo "[INFO] Extra Args: $EXTRA_ARGS"

# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_custom.py"

# 设置环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=$NPROC
export HF_ENDPOINT=https://hf-mirror.com

# 运行推理
torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    ${INFERENCE_SCRIPT} \
    --point_cloud ${POINT_CLOUD} \
    --output ${OUTPUT} \
    --model_path ${MODEL_PATH} \
    ${EXTRA_ARGS}
