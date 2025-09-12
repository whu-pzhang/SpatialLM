"""
已有数据结构如下：

data/HC3D/HC3D
├── scene_00000
│   ├── dxf
│   └── pcd
├── scene_00001
│   ├── dxf
│   └── pcd
├── scene_00002
│   ├── dxf
│   └── pcd
├── scene_00003
│   ├── dxf
│   └── pcd
├── scene_00004
│   ├── dxf
│   └── pcd
├── scene_00005
│   ├── dxf
│   └── pcd
├── scene_00006
│   ├── dxf
│   └── pcd
├── scene_00007
│   ├── dxf
│   └── pcd
├── scene_00008
│   ├── dxf
│   └── pcd
├── scene_00009
│   ├── dxf
│   └── pcd
└── scene_00010
    ├── dxf
    └── pcd


根据 dxf 目录下的 floorplan.json 文件中的 boundingBox 字段，对相应的las 点云文件进行
裁剪，将新的点云以 ply 格式保存到指定目录。
"""

import argparse
import json
import os

import laspy
import numpy as np
import open3d as o3d


def read_floorplan_json(json_path):
    """
    读取floorplan.json文件并提取boundingBox信息

    Args:
        json_path (str): floorplan.json文件路径

    Returns:
        tuple: (las_file_name, bounding_box)
               las_file_name: las文件名
               bounding_box: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取第一个frame的信息
    frame = data["frames"][0]

    # 获取las文件名
    las_file_name = frame["frameInfo"]["las_file_name"][0]
    las_file_name = os.path.basename(las_file_name)  # 只取文件名部分

    # 获取boundingBox
    bounding_box = frame["boundingBox"]

    return las_file_name, bounding_box


def load_las_file(las_path):
    """
    加载las点云文件，只提取坐标、颜色和强度信息

    Args:
        las_path (str): las文件路径

    Returns:
        tuple: (points, colors, intensity)
               points: 点云坐标数据 [N, 3] (x, y, z)
               colors: RGB颜色数据 [N, 3] (r, g, b)，如果没有颜色则为None
    """
    las_file = laspy.read(las_path)

    # 提取xyz坐标
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()

    # 尝试提取RGB颜色信息
    colors = None
    try:
        if (
            hasattr(las_file, "red")
            and hasattr(las_file, "green")
            and hasattr(las_file, "blue")
        ):
            scale = np.array(
                [las_file.red.max(), las_file.green.max(), las_file.blue.max()]
            )
            rgb = np.vstack((las_file.red, las_file.green, las_file.blue)).T
            colors = np.round(rgb / scale * 255.0).astype(np.uint8)
            # 使用更科学的方法将 uint16 的 RGB 值转换为 uint8
            # 线性映射: new_value = old_value * 255 / 65535
            # r = np.round(las_file.red * 255.0 / 65535.0).astype(np.uint8)
            # g = np.round(las_file.green * 255.0 / 65535.0).astype(np.uint8)
            # b = np.round(las_file.blue * 255.0 / 65535.0).astype(np.uint8)
            # colors = np.vstack((r, g, b)).transpose()
            print("  - 检测到RGB颜色信息")
        else:
            print("  - 未检测到RGB颜色信息")
    except Exception as e:
        print(f"  - 读取RGB颜色信息时出错: {str(e)}")
        colors = None

    return points, colors


def load_ply_file(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def crop_point_cloud(points, colors, bounding_box):
    """
    根据boundingBox裁剪点云

    Args:
        points (numpy.ndarray): 原始点云数据 [N, 3]
        colors (numpy.ndarray or None): 原始颜色数据 [N, 3]，如果为None则无颜色
        bounding_box (list): [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        tuple: (cropped_points, cropped_colors, cropped_intensity)
               cropped_points: 裁剪后的点云数据 [M, 3]
               cropped_colors: 裁剪后的颜色数据 [M, 3]，如果输入colors为None则返回None
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bounding_box

    # 创建掩码，筛选在boundingBox内的点
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )

    cropped_points = points[mask]
    cropped_colors = colors[mask] if colors is not None else None

    return cropped_points, cropped_colors


def downsample_point_cloud(points, colors, voxel_size=0.01):
    """
    使用体素下采样对点云进行抽稀

    Args:
        points (numpy.ndarray): 点云数据 [N, 3]
        colors (numpy.ndarray or None): 颜色数据 [N, 3]，如果为None则无颜色
        voxel_size (float): 体素大小，默认为0.01

    Returns:
        tuple: (downsampled_points, downsampled_colors)
               downsampled_points: 抽稀后的点云数据 [M, 3]
               downsampled_colors: 抽稀后的颜色数据 [M, 3]，如果输入colors为None则返回None
    """
    if len(points) == 0:
        return points, colors

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # 确保颜色在0-1范围内
        if colors.max() > 1.0:
            colors_norm = colors / 255.0
        else:
            colors_norm = colors
        pcd.colors = o3d.utility.Vector3dVector(colors_norm)

    # 执行体素下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # 提取下采样后的点和颜色
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors) if downsampled_pcd.has_colors() else None
    
    # 如果有颜色数据，将其转换回0-255范围
    if downsampled_colors is not None and downsampled_colors.max() <= 1.0:
        downsampled_colors = (downsampled_colors * 255.0).astype(np.uint8)

    return downsampled_points, downsampled_colors


def save_as_ply(points, colors, output_path):
    """
    将点云保存为ply格式，保留颜色信息，不对原始数据做任何处理

    Args:
        points (numpy.ndarray): 点云数据 [N, 3]
        colors (numpy.ndarray or None): 颜色数据 [N, 3]，如果为None则无颜色
        output_path (str): 输出ply文件路径
    """
    num_points = len(points)

    with open(output_path, "w") as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # 如果有颜色信息
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        # 写入点数据
        for i in range(num_points):
            # 写入坐标（保持原始精度）
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}")

            # 写入颜色（保持原始数值，不做任何处理）
            if colors is not None:
                f.write(f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}")

            f.write("\n")


def save_as_ply_open3d(points, colors, output_path):
    """
    使用Open3D将点云保存为ply格式，保留颜色信息

    Args:
        points (numpy.ndarray): 点云数据 [N, 3]
        colors (numpy.ndarray or None): 颜色数据 [N, 3]，如果为None则无颜色
        output_path (str): 输出ply文件路径
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        # 确保颜色在0-1范围内
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)


def process_single_scene(scene_path, output_dir, downsample=False, voxel_size=0.01):
    """
    处理单个场景

    Args:
        scene_path (str): 场景目录路径
        output_dir (str): 输出目录路径
        downsample (bool): 是否进行点云抽稀
        voxel_size (float): 抽稀时的体素大小

    Returns:
        bool: 处理是否成功
    """
    scene_name = os.path.basename(scene_path)
    print(f"正在处理场景: {scene_name}")

    # 构建文件路径
    json_path = os.path.join(scene_path, "dxf", "floorplan.json")
    pcd_dir = os.path.join(scene_path, "pcd")

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"警告: {json_path} 不存在，跳过该场景")
        return False

    if not os.path.exists(pcd_dir):
        print(f"警告: {pcd_dir} 不存在，跳过该场景")
        return False

    try:
        # 读取floorplan.json
        las_file_name, bounding_box = read_floorplan_json(json_path)
        print(f"  - las文件: {las_file_name}")
        print(f"  - boundingBox: {bounding_box}")

        # 构建las文件路径
        las_path = os.path.join(pcd_dir, las_file_name)

        if not os.path.exists(las_path):
            print(f"警告: {las_path} 不存在，跳过该场景")
            return False

        # 加载点云
        print("  - 正在加载点云...")
        points, colors = load_las_file(las_path)
        print(f"  - 原始点云包含 {len(points)} 个点")

        # 裁剪点云
        print("  - 正在裁剪点云...")
        cropped_points, cropped_colors = crop_point_cloud(points, colors, bounding_box)
        print(f"  - 裁剪后点云包含 {len(cropped_points)} 个点")

        if len(cropped_points) == 0:
            print("警告: 裁剪后点云为空，跳过该场景")
            return False

        # 如果需要抽稀，则进行抽稀操作
        if downsample:
            print(f"  - 正在进行点云抽稀 (体素大小: {voxel_size})...")
            cropped_points, cropped_colors = downsample_point_cloud(cropped_points, cropped_colors, voxel_size)
            print(f"  - 抽稀后点云包含 {len(cropped_points)} 个点")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存为ply文件
        # ply_filename = las_file_name.replace(".las", "_cropped.ply")
        ply_filename = f"{scene_name}.ply"
        output_path = os.path.join(output_dir, ply_filename)

        print("  - 正在保存ply文件...")
        save_as_ply_open3d(cropped_points, cropped_colors, output_path)
        print(f"  - 已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"错误: 处理场景 {scene_name} 时出现异常: {str(e)}")
        return False


def process_hc3d_dataset(data_root, output_dir, downsample=False, voxel_size=0.01):
    """
    批量处理HC3D数据集

    Args:
        data_root (str): HC3D数据集根目录路径
        output_dir (str): 输出目录路径
        downsample (bool): 是否进行点云抽稀
        voxel_size (float): 抽稀时的体素大小
    """
    print(f"开始处理HC3D数据集: {data_root}")
    print(f"输出目录: {output_dir}")
    if downsample:
        print(f"点云抽稀已启用，体素大小: {voxel_size}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有场景目录
    scene_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item.startswith("scene_"):
            scene_dirs.append(item_path)

    scene_dirs.sort()  # 按名称排序

    print(f"找到 {len(scene_dirs)} 个场景")

    # 处理每个场景
    success_count = 0
    for scene_path in scene_dirs:
        if process_single_scene(scene_path, output_dir, downsample, voxel_size):
            success_count += 1
        print()  # 空行分隔

    print(f"处理完成! 成功处理 {success_count}/{len(scene_dirs)} 个场景")


def main():
    parser = argparse.ArgumentParser(
        description="处理HC3D数据集，根据boundingBox裁剪点云并保存为ply格式"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/HC3D/HC3D", help="HC3D数据集根目录路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/HC3D/processed/pcd", help="输出目录路径"
    )
    parser.add_argument(
        "--downsample", action="store_true", help="是否进行点云抽稀"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.01, help="抽稀时的体素大小"
    )

    args = parser.parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录 {args.data_root} 不存在")
        return

    # 处理数据集
    process_hc3d_dataset(args.data_root, args.output_dir, args.downsample, args.voxel_size)


if __name__ == "__main__":
    main()