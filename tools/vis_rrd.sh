#!/usr/bin/env bash
# 检查是否提供了至少一个layout路径参数
if [ $# -lt 1 ]; then
    echo "使用方法: $0 <layout_path1> [<layout_path2> ...]"
    echo "示例: $0 data/*_test_sft_g640/scene_03250.txt data/*_test_sft_g640/scene_03251.txt"
    echo "也可以使用通配符批量处理: $0 data/*_test_sft_g640/scene_*.txt"
    exit 1
fi

# 循环处理每个输入的layout文件
for layout_path in "$@"; do
    # 检查文件是否存在
    if [ ! -f "$layout_path" ]; then
        echo "警告: 文件 $layout_path 不存在，跳过处理"
        continue
    fi

    # 提取场景名称（假设文件名格式为scene_xxxxxx.txt）
    scene_name=$(basename "$layout_path" .txt)
    dir_name=$(dirname "$layout_path")

    # 构建其他参数的路径
    # point_cloud_path="data/structured3d-spatiallm/pcd_test/${scene_name}.ply"
    point_cloud_path="data/HC3D/20251127/pcd/${scene_name}.ply"
    save_dir="${dir_name}/rrd"
    save_path="${save_dir}/${scene_name}.rrd"

    # 检查点云文件是否存在
    if [ ! -f "$point_cloud_path" ]; then
        echo "警告: 点云文件 $point_cloud_path 不存在，跳过处理 $layout_path"
        continue
    fi

    # 创建保存目录（如果不存在）
    mkdir -p "$save_dir"

    # 执行可视化命令
    echo "正在处理: $layout_path"
    echo "执行命令: python visualize.py --point_cloud \"$point_cloud_path\" --layout \"$layout_path\" --save \"$save_path\" "
    python3 visualize.py --point_cloud "$point_cloud_path" --layout "$layout_path" --save "$save_path" --hide-labels
    
    # 检查命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "处理成功: $scene_name"
    else
        echo "处理失败: $scene_name"
    fi
    echo "----------------------------------------"
done

echo "所有文件处理完毕"
