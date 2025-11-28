import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

from spatiallm.layout import Layout


def random_within_delta(
    value: float,
    delta: float,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    在 [value - delta, value + delta] 范围内生成随机数。
    - clamp_min / clamp_max: 可选的全局上下界（在取样后裁剪）
    - seed: 可选，设置随机种子以复现结果
    """
    if seed is not None:
        random.seed(seed)
    low = value - delta
    high = value + delta

    out = random.uniform(low, high)
    if clamp_min is not None:
        out = max(out, clamp_min)
    if clamp_max is not None:
        out = min(out, clamp_max)
    return out


def random_within_percent(value: float, percent: float, seed: Optional[int] = None):
    """
    percent = 0.1 表示 ±10% 范围，即 [value*(1-0.1), value*(1+0.1)]
    """
    delta = abs(value) * percent
    return random_within_delta(value, delta, seed=seed)


def process_scene(
    layout_path: Path,
    pcd_path: Path,
    z_min: float = 0.5,
    z_max: float = 2.2,
):
    """ """
    layout = Layout(layout_path.read_text())
    layout.clip_z(world_min=z_min, world_max=z_max)
    # print(layout.to_language_string())

    # process pcd
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_m = np.asarray(pcd.points) / 1000.0
    mask = (points_m[:, 2] >= z_min) & (points_m[:, 2] <= z_max)

    clipped_points = points_m[mask]

    clipped_pcd = o3d.geometry.PointCloud()
    clipped_pcd.points = o3d.utility.Vector3dVector(clipped_points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[mask]
        clipped_pcd.colors = o3d.utility.Vector3dVector(colors)

    return clipped_pcd, layout


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="创建自定义Structure3D数据")

    # 输入路径参数
    parser.add_argument(
        "--stru3d_root",
        type=str,
        default=r"Z:\05_DL_dataset_wuhan\03_开源数据集\structure_3D\Panorama",
        help="Structure3D数据集根目录",
    )
    parser.add_argument(
        "--stru3d_spatiallm_root",
        type=str,
        default=r"Z:\09_LLM\structured3d-spatiallm",
        help="Structure3D SpatialLM数据根目录",
    )

    # 输出路径参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\data\huace_pcd\stru3d_clipped",
        help="输出目录根路径",
    )

    # Z轴裁剪参数
    parser.add_argument("--z_min", type=float, default=0.5, help="Z轴最小裁剪值(米)")
    parser.add_argument("--z_max", type=float, default=2.2, help="Z轴最大裁剪值(米)")
    parser.add_argument(
        "--delta_z", type=float, default=0.2, help="Z轴随机变化范围(米)"
    )

    # 处理限制参数
    parser.add_argument(
        "--max_scenes", type=int, default=0, help="最大处理场景数量(0表示处理所有场景)"
    )

    # 随机种子参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子(用于复现结果)")

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")

    # 设置路径
    stru3d_root = Path(args.stru3d_root)
    stru3d_spatiallm_root = Path(args.stru3d_spatiallm_root)
    out_pcd_dir = Path(args.output_dir) / "pcd"
    out_layout_dir = Path(args.output_dir) / "layout"

    # 创建输出目录
    out_pcd_dir.mkdir(parents=True, exist_ok=True)
    out_layout_dir.mkdir(parents=True, exist_ok=True)

    # 获取场景列表
    layout_list = [f for f in stru3d_spatiallm_root.joinpath("layout").glob("*.txt")]
    scene_list = [f.stem for f in layout_list]

    # 初始化计数器和Z轴参数
    cnt = 0
    z_min = args.z_min
    z_max = args.z_max

    print(
        f"开始处理数据，最大场景数量: {args.max_scenes if args.max_scenes > 0 else '无限制'}"
    )

    for part in stru3d_root.iterdir():
        if part.is_file():
            continue
        scenes = [d for d in part.joinpath("Structured3D").iterdir() if d.is_dir()]
        scenes = [s for s in scenes if s.name in scene_list]

        # 为场景处理循环添加进度条
        for scene in tqdm(scenes, desc=f"处理 {part.name} 中的场景"):
            scene_name = scene.name
            pcd_path = scene.joinpath("point_cloud.ply")
            layout_path = stru3d_spatiallm_root.joinpath("layout", f"{scene_name}.txt")

            # 随机化Z轴参数
            current_z_min = random_within_delta(z_min, args.delta_z, seed=args.seed)
            current_z_max = random_within_delta(z_max, args.delta_z, seed=args.seed)

            pcd, layout = process_scene(
                layout_path, pcd_path, z_min=current_z_min, z_max=current_z_max
            )
            # 保存数据
            out_pcd_path = out_pcd_dir / f"{scene_name}.ply"
            o3d.io.write_point_cloud(str(out_pcd_path), pcd)
            out_layout_path = out_layout_dir / f"{scene_name}.txt"
            out_layout_path.write_text(layout.to_language_string())

            cnt += 1
            # 检查是否达到最大处理数量
            if args.max_scenes > 0 and cnt >= args.max_scenes:
                print(f"已达到最大处理场景数量 {args.max_scenes}，停止处理")
                break

        # 如果已经处理了足够的场景，退出外层循环
        if args.max_scenes > 0 and cnt >= args.max_scenes:
            break

    print(f"处理完成，共处理了 {cnt} 个场景")


if __name__ == "__main__":
    main()
