import argparse
import os
import re
import time
import warnings
from pathlib import Path
from threading import Thread

import laspy
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed,
)

from spatiallm import Layout
from spatiallm.pcd import Compose, cleanup_pcd, get_points_and_colors, load_o3d_pcd
from tools.txt2dxf import txt2dxf  # noqa
from tools.txt2dxf_3d import Txt2DxfConverter  # noqa

warnings.filterwarnings("ignore")  # noqa


def format_timing_message(record):
    """自定义日志格式化函数，为耗时信息添加颜色高亮"""
    message = record["message"]
    level = record["level"].name
    
    # 匹配耗时信息的正则表达式
    timing_pattern = r'(\d+\.?\d*)(s|ms)'
    
    def colorize_timing(match):
        value = float(match.group(1))
        unit = match.group(2)
        
        # 根据时长和单位选择颜色
        if unit == 's':  # 秒
            if value >= 10:
                return f"<red><bold>{value}{unit}</bold></red>"
            elif value >= 1:
                return f"<yellow><bold>{value}{unit}</bold></yellow>"
            else:
                return f"<green><bold>{value}{unit}</bold></green>"
        else:  # 毫秒
            if value >= 1000:
                return f"<yellow><bold>{value}{unit}</bold></yellow>"
            else:
                return f"<green><bold>{value}{unit}</bold></green>"
    
    # 替换耗时信息
    message = re.sub(timing_pattern, colorize_timing, message)
    
    # 为特定关键词添加颜色
    keywords = {
        'completed': '<green>completed</green>',
        'failed': '<red>failed</red>',
        'error': '<red>error</red>',
        'success': '<green>success</green>',
        'warning': '<yellow>warning</yellow>',
        'loading': '<blue>loading</blue>',
        'processing': '<cyan>processing</cyan>'
    }
    
    for keyword, colored in keywords.items():
        message = message.replace(keyword, colored)
    
    # 根据日志级别设置颜色
    level_colors = {
        'DEBUG': '<cyan>DEBUG</cyan>',
        'INFO': '<blue>INFO</blue>',
        'WARNING': '<yellow>WARNING</yellow>',
        'ERROR': '<red>ERROR</red>',
        'CRITICAL': '<red><bold>CRITICAL</bold></red>'
    }
    
    colored_level = level_colors.get(level, level)
    
    # 返回格式化的消息
    return f"{record['time']:YYYY-MM-DD HH:mm:ss} | {colored_level} | {message}\n"


def configure_logger(debug_mode=False):
    """配置日志系统"""
    logger.remove()  # Remove default handler
    
    # 文件日志始终记录DEBUG级别
    logger.add(
        "spatiallm_infer_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        encoding="utf-8"
    )
    
    # 控制台日志根据debug模式调整级别
    console_level = "DEBUG" if debug_mode else "INFO"
    logger.add(
        lambda msg: print(msg, end=""),
        level=console_level,
        format=format_timing_message,
        colorize=True
    )
    
    if debug_mode:
        logger.info("🐛 Debug mode enabled - showing detailed logging information")
    
    return logger


DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}


def load_pcd_file(pcd_path):
    logger.info(f"Loading point cloud file: {pcd_path}")
    suffix = Path(pcd_path).suffix
    
    try:
        if suffix == ".ply":
            logger.debug("Loading PLY file using Open3D")
            pcd = load_o3d_pcd(pcd_path)
        elif suffix in [".las", ".laz"]:
            logger.debug("Loading LAS/LAZ file")
            pcd = load_las_file(pcd_path)
        else:
            logger.error(f"Unsupported point cloud file format: {suffix}")
            raise ValueError("Unsupported point cloud file format")
        
        logger.info(f"Successfully loaded point cloud with {len(pcd.points)} points")
        return pcd
    except Exception as e:
        logger.error(f"Failed to load point cloud file {pcd_path}: {str(e)}")
        raise


def load_las_file(las_path):
    """
    load las file and convert to o3d.geometry.PointCloud

    Args:
        las_path (str): las文件路径
    """
    logger.debug(f"Reading LAS file: {las_path}")
    las_file = laspy.read(las_path)
    logger.debug(f"LAS file contains {len(las_file.points)} points")

    # 提取xyz坐标
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    logger.debug(f"Extracted {points.shape[0]} XYZ coordinates")

    # 尝试提取RGB颜色信息
    colors = None
    try:
        if (
            hasattr(las_file, "red")
            and hasattr(las_file, "green")
            and hasattr(las_file, "blue")
        ):
            # 获取最大值用于归一化
            max_vals = [
                las_file.red.max() if las_file.red.max() > 0 else 1,
                las_file.green.max() if las_file.green.max() > 0 else 1,
                las_file.blue.max() if las_file.blue.max() > 0 else 1,
            ]
            scale = np.array(max_vals)

            # 提取RGB值并归一化到0-1范围
            rgb = np.vstack((las_file.red, las_file.green, las_file.blue)).T
            colors = rgb / scale
        else:
            logger.warning("未检测到RGB颜色信息")
    except Exception as e:
        logger.error(f"读取RGB颜色信息时出错: {str(e)}")
        colors = None

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果有颜色信息，则添加到点云中
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ===============
def preprocess_point_cloud(points, colors, grid_size, num_bins):
    logger.debug(f"Preprocessing point cloud with {points.shape[0]} points, grid_size={grid_size}, num_bins={num_bins}")
    
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
    
    logger.debug("Applying point cloud transformations")
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
    logger.debug(f"After transformation: coord shape={coord.shape}, xyz shape={xyz.shape}, rgb shape={rgb.shape}")
    
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    result = torch.as_tensor(np.stack([point_cloud], axis=0))
    logger.debug(f"Final preprocessed tensor shape: {result.shape}")
    
    return result


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
):
    generation_start = time.time()
    logger.info(f"Starting layout generation with detect_type={detect_type}, seed={seed}")
    logger.debug(f"Generation parameters: top_k={top_k}, top_p={top_p}, temperature={temperature}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")
    
    if seed >= 0:
        logger.debug(f"Setting random seed to {seed}")
        set_seed(seed)

    # load the code template
    template_start = time.time()
    logger.debug(f"Loading code template from: {code_template_file}")
    try:
        with open(code_template_file, "r") as f:
            code_template = f.read()
        template_time = time.time() - template_start
        logger.debug(f"📄 Code template loaded in {template_time:.3f}s")
    except Exception as e:
        logger.error(f"Failed to load code template from {code_template_file}: {str(e)}")
        raise

    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    logger.info(f"Task prompt: {task_prompt}")

    prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt} The reference code is as followed: {code_template}"

    # prepare the conversation data
    tokenize_start = time.time()
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
    tokenize_time = time.time() - tokenize_start
    logger.debug(f"🔤 Tokenization completed in {tokenize_time:.3f}s")

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

    inference_start = time.time()
    logger.info("🤖 Generating layout...")
    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
        # 只记录非空且有意义的文本块，去除换行符
        if text and text.strip():
            clean_text = text.replace('\n', '\\n').replace('\r', '\\r')
            logger.info(f"Generated layout chunk: {clean_text}")
    inference_time = time.time() - inference_start
    logger.info(f"🎯 Layout generation completed in {inference_time:.2f}s!")

    layout_parse_start = time.time()
    layout_str = "".join(generate_texts)
    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    layout_parse_time = time.time() - layout_parse_start
    total_generation_time = time.time() - generation_start
    logger.debug(f"📊 Layout parsing completed in {layout_parse_time:.3f}s | ⏱️ Total generation time: {total_generation_time:.2f}s")
    return layout


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
        default="weights/spatiallm-0.5b-sft",
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
        default=0.1,
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
        default=42,
        help="The seed to use during inference, negative value means no seed",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode with verbose logging and detailed information",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    program_start_time = time.time()
    args = parse_args()
    
    # 配置日志系统
    logger = configure_logger(debug_mode=args.debug)

    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Point cloud input: {args.point_cloud}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Detection type: {args.detect_type}")
    
    if args.debug:
        logger.debug(f"🔧 Debug mode parameters:")
        logger.debug(f"  - Code template: {args.code_template_file}")
        logger.debug(f"  - Top-k: {args.top_k}")
        logger.debug(f"  - Top-p: {args.top_p}")
        logger.debug(f"  - Temperature: {args.temperature}")
        logger.debug(f"  - Num beams: {args.num_beams}")
        logger.debug(f"  - Inference dtype: {args.inference_dtype}")
        logger.debug(f"  - No cleanup: {args.no_cleanup}")
        logger.debug(f"  - Seed: {args.seed}")
        logger.debug(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"  - CUDA device count: {torch.cuda.device_count()}")
            logger.debug(f"  - Current CUDA device: {torch.cuda.current_device()}")
            logger.debug(f"  - CUDA device name: {torch.cuda.get_device_name()}")

    # load the model
    logger.info("Loading model and tokenizer...")
    model_load_start = time.time()
    try:
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer_time = time.time() - tokenizer_start
        logger.info(f"⚡ Tokenizer loaded successfully in {tokenizer_time:.2f}s")
        
        model_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=getattr(torch, args.inference_dtype)
        )
        model_time = time.time() - model_start
        logger.info(f"🧠 Model loaded with dtype {args.inference_dtype} in {model_time:.2f}s")

        cuda_start = time.time()
        model.to("cuda")
        model.set_point_backbone_dtype(torch.float32)
        model.eval()
        cuda_time = time.time() - cuda_start
        total_load_time = time.time() - model_load_start
        logger.info(f"🚀 Model moved to CUDA in {cuda_time:.2f}s | 📊 Total load time: {total_load_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # number of bins used for discretization
    num_bins = model.config.point_config["num_bins"]
    logger.debug(f"Using {num_bins} bins for discretization")

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
        logger.info("Processing single point cloud file")
    else:
        point_cloud_files = [
            f.as_posix()
            for f in Path(args.point_cloud).glob("*")
            if f.suffix in [".las", ".ply"]
        ]
        logger.info(f"Found {len(point_cloud_files)} point cloud files to process")

    for point_cloud_file in point_cloud_files:
        file_start_time = time.time()
        logger.info(f"Processing point cloud file: {point_cloud_file}")

        try:
            # Load point cloud
            load_start = time.time()
            point_cloud = load_pcd_file(point_cloud_file)
            load_time = time.time() - load_start
            logger.info(f"📁 Point cloud loaded in {load_time:.2f}s")
            
            grid_size = Layout.get_grid_size(num_bins)
            logger.debug(f"Grid size: {grid_size}")

            if not args.no_cleanup:
                cleanup_start = time.time()
                logger.debug("Cleaning up point cloud")
                point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)
                cleanup_time = time.time() - cleanup_start
                logger.debug(f"🧹 Point cloud cleanup completed in {cleanup_time:.2f}s")

            extract_start = time.time()
            points, colors = get_points_and_colors(point_cloud)
            min_extent = np.min(points, axis=0)
            extract_time = time.time() - extract_start
            logger.debug(f"📐 Point cloud extent extracted in {extract_time:.2f}s: min={min_extent}")

            # Preprocess point cloud
            preprocess_start = time.time()
            input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
            preprocess_time = time.time() - preprocess_start
            logger.info(f"⚙️ Point cloud preprocessing completed in {preprocess_time:.2f}s")

            # Generate the layout
            generation_start = time.time()
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
            )
            generation_time = time.time() - generation_start
            
            layout.translate(min_extent)
            pred_language_string = layout.to_language_string()

            # Save the output
            save_start = time.time()
            if os.path.splitext(args.output)[-1]:
                logger.info(f"Saving output to file: {args.output}")
                with open(args.output, "w") as f:
                    f.write(pred_language_string)
                stem = Path(args.output).stem
                
                # Convert to 2D DXF
                dxf_2d_start = time.time()
                dxf_2d_file = Path(args.output).parent.joinpath(f"{stem}_2d.dxf")
                logger.debug(f"Converting to 2D DXF: {dxf_2d_file}")
                txt2dxf(input_file=args.output, output_file=dxf_2d_file, verbose=False)
                dxf_2d_time = time.time() - dxf_2d_start
                
                # Convert to 3D DXF
                dxf_3d_start = time.time()
                dxf_3d_file = Path(args.output).parent.joinpath(f"{stem}_3d.dxf")
                logger.debug(f"Converting to 3D DXF: {dxf_3d_file}")
                converter = Txt2DxfConverter(
                    input_file=args.output, output_file=dxf_3d_file
                )
                converter.convert()
                dxf_3d_time = time.time() - dxf_3d_start
            else:
                output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
                os.makedirs(args.output, exist_ok=True)
                output_filepath = os.path.join(args.output, output_filename)
                logger.info(f"Saving output to directory: {output_filepath}")
                stem = Path(output_filepath).stem
                with open(output_filepath, "w") as f:
                    f.write(pred_language_string)
                
                # Convert to 2D DXF
                dxf_2d_start = time.time()
                dxf_2d_file = Path(output_filepath).parent.joinpath(f"{stem}_2d.dxf")
                logger.debug(f"Converting to 2D DXF: {dxf_2d_file}")
                txt2dxf(
                    input_file=output_filepath, output_file=str(dxf_2d_file), verbose=False
                )
                dxf_2d_time = time.time() - dxf_2d_start
                
                # Convert to 3D DXF
                dxf_3d_start = time.time()
                dxf_3d_file = Path(output_filepath).parent.joinpath(f"{stem}_3d.dxf")
                logger.debug(f"Converting to 3D DXF: {dxf_3d_file}")
                converter = Txt2DxfConverter(
                    input_file=output_filepath, output_file=dxf_3d_file
                )
                converter.convert()
                dxf_3d_time = time.time() - dxf_3d_start
            
            save_time = time.time() - save_start
            file_total_time = time.time() - file_start_time
            logger.info(f"💾 File saved and converted in {save_time:.2f}s | 📐 2D DXF: {dxf_2d_time:.2f}s | 🏗️ 3D DXF: {dxf_3d_time:.2f}s")
            logger.info(f"✅ Total processing time for {os.path.basename(point_cloud_file)}: {file_total_time:.2f}s")
                
        except Exception as e:
            file_error_time = time.time() - file_start_time
            logger.error(f"❌ Failed to process {point_cloud_file} after {file_error_time:.2f}s: {str(e)}")
            continue
            
    program_total_time = time.time() - program_start_time
    logger.info(f"🎉 All point cloud files processed successfully in {program_total_time:.2f}s")
