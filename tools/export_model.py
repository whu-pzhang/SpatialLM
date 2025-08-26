import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spatiallm import Layout  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description="导出 SpatialLM 模型")
    parser.add_argument(
        "--src",
        type=str,
        default="saves/spatiallm-qwen-0.5b-sft/checkpoint-16050",
        help="源模型路径 (默认: saves/spatiallm-qwen-0.5b-sft/checkpoint-16050)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="exports/spatiallm-qwen-0.5b-sft-fp32",
        help="目标导出路径 (默认: exports/spatiallm-qwen-0.5b-sft-fp32)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="模型数据类型 (默认: float32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备 (默认: cuda)"
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="不使用安全序列化 (.safetensors)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    print(f"📂 源模型路径: {args.src}")
    print(f"📁 目标路径: {args.dst}")
    print(f"🔢 数据类型: {args.dtype}")
    print(f"💻 设备: {args.device}")
    
    # 加载模型和分词器
    print("🔄 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.src)
    
    print("🔄 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch_dtype)
    model.to(args.device)
    model.set_point_backbone_dtype(torch_dtype)
    
    # 保存模型
    print("💾 保存模型...")
    safe_serialization = not args.no_safe_serialization
    model.save_pretrained(args.dst, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(args.dst)
    
    serialization_type = ".safetensors" if safe_serialization else ".bin"
    print(f"✅ 模型已导出到: {args.dst} (格式: {serialization_type})")


if __name__ == "__main__":
    main()
