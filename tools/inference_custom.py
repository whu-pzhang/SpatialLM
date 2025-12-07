import argparse
import os
import random
from pathlib import Path
from threading import Thread
import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed,
)

from spatiallm import Layout
from spatiallm.constants import POINT_E_TOKEN, POINT_PAD_TOKEN, POINT_S_TOKEN
from spatiallm.pcd import Compose, cleanup_pcd, get_points_and_colors, load_o3d_pcd
from spatiallm.prompts import DETECT_TYPE_PROMPT


def setup_distributed():
    """Setup distributed inference environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=60),
        )
        torch.cuda.set_device(local_rank)
        print(f"Initialized process {rank}/{world_size} (local_rank: {local_rank})")
        return rank, world_size, local_rank
    else:
        print("Not using distributed mode")
        return 0, 1, 0


def set_deterministic_seed(seed):
    """设置所有相关库的随机种子以确保完全可重复性"""
    import random

    import numpy as np
    import torch

    # 设置 Python 随机种子
    random.seed(seed)

    # 设置 NumPy 随机种子
    np.random.seed(seed)

    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置 CUDA 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    code_template_file,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    seed=-1,
    max_new_tokens=4096,
    detect_type="all",
    categories=[],
    verbose=True,
):
    if seed >= 0:
        set_seed(seed)

    # load the code template
    with open(code_template_file, "r") as f:
        code_template = f.read()

    task_prompt = random.choice(DETECT_TYPE_PROMPT[detect_type])
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))

    if verbose:
        print("Task prompt: ", task_prompt)

    prompt = f"{POINT_S_TOKEN}{POINT_PAD_TOKEN}{POINT_E_TOKEN}{task_prompt} The reference code is as followed: {code_template}"

    # prepare the conversation data
    if model.config.model_type == "spatiallm_qwen":
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    # Set pad_token_id for generation
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    generate_kwargs = dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "point_clouds": point_cloud,
        },
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        pad_token_id=pad_token_id,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    if verbose:
        print("Generating layout...\n")
    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
        if verbose:
            print(text, end="", flush=True)
    if verbose:
        print("\nDone!")

    layout_str = "".join(generate_texts)
    # The layout_str may contain special tokens <int>.
    # Layout.from_str() has been updated to parse these tokens into integers.
    layout = Layout(layout_str)

    # After parsing, we need to undiscretize and unnormalize the integer coordinates back to physical world coordinates.
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    return layout


def get_pcd_list(csv_file):
    """
    Read a list of point cloud files from a CSV file.

    Args:
        data_root: Root directory for relative paths (not used in current implementation)
        csv_file: Path to CSV file containing point cloud file paths

    Returns:
        List of point cloud file paths
    """
    df = pd.read_csv(csv_file)
    # Check if 'pcd' column exists, otherwise use the first column
    pcd_list = df["pcd"].tolist()
    pcd_list = [Path(f).name for f in pcd_list]

    return pcd_list


def parse_args():
    parser = argparse.ArgumentParser("SpatialLM inference script")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file or a folder containing multiple point cloud files",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        required=False,
        help="Path to a CSV file containing a list of point cloud files to process. The CSV should have a 'pcd' column or use the first column for file paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output layout txt file or a folder to save multiple layout txt files",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM-Llama-1B",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        type=str,
        default="arch",
        choices=["all", "arch", "object"],
        help="The type of indoor elements to detect. all: (wall, door, window, box), arch: (wall, door, window), object: (box)",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        choices=[
            "sofa",
            "chair",
            "dining_chair",
            "bar_chair",
            "stool",
            "bed",
            "pillow",
            "wardrobe",
            "nightstand",
            "tv_cabinet",
            "wine_cabinet",
            "bathroom_cabinet",
            "shoe_cabinet",
            "entrance_cabinet",
            "decorative_cabinet",
            "washing_cabinet",
            "wall_cabinet",
            "sideboard",
            "cupboard",
            "coffee_table",
            "dining_table",
            "side_table",
            "dressing_table",
            "desk",
            "integrated_stove",
            "gas_stove",
            "range_hood",
            "micro-wave_oven",
            "sink",
            "stove",
            "refrigerator",
            "hand_sink",
            "shower",
            "shower_room",
            "toilet",
            "tub",
            "illumination",
            "chandelier",
            "floor-standing_lamp",
            "wall_decoration",
            "painting",
            "curtain",
            "carpet",
            "plants",
            "potted_bonsai",
            "tv",
            "computer",
            "air_conditioner",
            "washing_machine",
            "clothes_rack",
            "mirror",
            "bookcase",
            "cushion",
            "bar",
            "screen",
            "combination_sofa",
            "dining_table_combination",
            "leisure_table_and_chair_combination",
            "multifunctional_combination_bed",
        ],
        help="A list of categories of objects to detect. If not specified, all categories will be detected.",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default="code_template.txt",
        help="Path to the code template file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="The number of highest probability vocabulary tokens to keep for top-k filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="bfloat16",
        help="The torch dtype to use for inference, bfloat16 or float32",
    )
    parser.add_argument(
        "--no_cleanup",
        default=False,
        action="store_true",
        help="Whether to not cleanup the point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The seed to use during inference, negative value means no seed",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.file_list and not os.path.isfile(args.file_list):
        raise FileNotFoundError(f"File list not found: {args.file_list}")

    # Initialize distributed environment
    rank, world_size, local_rank = setup_distributed()

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.inference_dtype),
        low_cpu_mem_usage=True,
    )
    # Use local_rank for device placement
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # number of bins used for discretization
    num_bins = model.config.point_config["num_bins"]

    # Get the list of point cloud files to process
    if args.file_list:
        # Read from CSV file list
        point_cloud_files = get_pcd_list(args.file_list)
        # Convert relative paths to absolute paths if needed
        point_cloud_files = [
            str(Path(args.point_cloud).joinpath(pcd)) for pcd in point_cloud_files
        ]
    else:
        # Check if the input is a single point cloud file or a folder containing multiple point cloud files
        point_cloud_path = Path(args.point_cloud)
        if point_cloud_path.is_file():
            point_cloud_files = [str(point_cloud_path)]
        else:
            point_cloud_files = [str(p) for p in point_cloud_path.glob("*.ply")]

    # Sort files to ensure deterministic order across processes
    point_cloud_files.sort()

    # Split files among processes
    if world_size > 1:
        total_files = len(point_cloud_files)
        files_per_rank = total_files // world_size
        remainder = total_files % world_size

        start_idx = rank * files_per_rank + min(rank, remainder)
        end_idx = start_idx + files_per_rank + (1 if rank < remainder else 0)

        point_cloud_files = point_cloud_files[start_idx:end_idx]
        print(
            f"Rank {rank}: Processing {len(point_cloud_files)} files (from index {start_idx} to {end_idx})"
        )

    # Only show progress bar on rank 0 or if not distributed
    if rank == 0 or world_size == 1:
        iterator = tqdm(point_cloud_files)
    else:
        iterator = point_cloud_files

    # Calculate total files processed by this rank for progress logging
    total_files_rank = len(point_cloud_files)

    for i, point_cloud_file in enumerate(iterator):
        # Log progress for non-zero ranks periodically
        if world_size > 1 and rank != 0 and (i + 1) % 10 == 0:
            print(f"Rank {rank}: Processed {i + 1}/{total_files_rank} files")

        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        grid_size = Layout.get_grid_size(num_bins)

        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # generate the layout
        layout = generate_layout(
            model,
            input_pcd,
            tokenizer,
            args.code_template_file,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=args.num_beams,
            seed=args.seed,
            detect_type=args.detect_type,
            categories=args.category,
            verbose=(
                world_size == 1
            ),  # Only print verbose output in single process mode
        )
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # check if the output path is a file or directory
        output_path = Path(args.output)
        if output_path.suffix:
            with open(output_path, "w") as f:
                f.write(pred_language_string)
        else:
            output_filename = Path(point_cloud_file).stem + ".txt"
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / output_filename, "w") as f:
                f.write(pred_language_string)
