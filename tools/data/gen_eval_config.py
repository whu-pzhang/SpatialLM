"""
指定模型预测路径和真值路径，生成用于评测程序的配置文件，文件示例内容如下，每一行用逗号分隔
，左边为真值路径，右边为预测路径：

Z:\01_3D-FAVP\handheld_scanner_Data\S20\188guojishequ_res_ff_S20_0\Annotations\floorplan.dxf,D:\data\huace_pcd\HC3D_20251103\all_pred_g1280_1107\dxf\scene_00001_0.02.dxf
Z:\01_3D-FAVP\handheld_scanner_Data\S20\188guojishequ_res_ff_S20_3\Annotations\floorplan.dxf,D:\data\huace_pcd\HC3D_20251103\all_pred_g1280_1107\dxf\scene_00004_0.02.dxf
Z:\01_3D-FAVP\handheld_scanner_Data\S20\188guojishequ_res_ff_S20_6\Annotations\floorplan.dxf,D:\data\huace_pcd\HC3D_20251103\all_pred_g1280_1107\dxf\scene_00007_0.02.dxf

预测和真值的对应关系，可以从 data_info 文件中读取，该文件为 csv 格式，示例如下：

data_name,place,house_type,device,furnish_type,path,scene_id
188guojishequ_res_ff_S20_0,188guojishequ,res,S20,ff,Z:\01_3D-FAVP\handheld_scanner_Data\S20\188guojishequ_res_ff_S20_0,00001
188guojishequ_res_ff_S20_1,188guojishequ,res,S20,ff,Z:\01_3D-FAVP\handheld_scanner_Data\S20\188guojishequ_res_ff_S20_1,00002


用户需制定模型预测路径和生成的配置文件路径，具体执行步骤如下：

1. 读取 data_root 下的 val.csv 文件，获取验证数据对应的 scene_id
2. 获取用户指定预测路径下的 dxf 文件列表
3. 读取 data_info 文件，获取 scene_id 对应的真值路径
4. 生成配置文件
"""

import argparse
from pathlib import Path

import pandas as pd


def read_val_csv(data_root):
    """
    读取 data_root 下的 val.csv 文件，获取验证数据对应的 scene_id

    Args:
        data_root (str): 数据根目录路径

    Returns:
        list: 验证数据的 scene_id 列表
    """
    val_csv_path = Path(data_root) / "val.csv"

    if not val_csv_path.exists():
        print(f"警告: 未找到 {val_csv_path} 文件")
        return []

    try:
        df = pd.read_csv(val_csv_path)
        # 从 id 列中提取 scene_id
        scene_ids = df["id"].tolist()
        print(f"从 {val_csv_path} 中读取到 {len(scene_ids)} 个验证数据")
        return scene_ids
    except Exception as e:
        print(f"读取 {val_csv_path} 时出错: {e}")
        return []


def get_pred_dxf_files(pred_path):
    """
    获取用户指定预测路径下的 dxf 文件列表

    Args:
        pred_path (str): 预测文件路径

    Returns:
        dict: scene_id 到预测文件路径的映射
    """
    pred_path = Path(pred_path)

    if not pred_path.exists():
        print(f"错误: 预测路径 {pred_path} 不存在")
        return {}

    # 获取所有 dxf 文件
    dxf_files = list(pred_path.glob("*.dxf"))

    # 解析文件名获取 scene_id
    scene_id_to_pred = {}
    for dxf_file in dxf_files:
        # 文件名格式: scene_XXXXX_0.02.dxf
        filename = dxf_file.name
        try:
            parts = filename.split("_")
            if len(parts) >= 2:
                scene_id = parts[1]  # 提取 scene_id 的数字部分
                # 保存完整的 scene_id 格式 (scene_XXXXX)
                full_scene_id = f"scene_{scene_id}"
                scene_id_to_pred[full_scene_id] = str(dxf_file.absolute())
        except Exception as e:
            print(f"解析文件名 {filename} 时出错: {e}")

    print(f"从 {pred_path} 中找到 {len(scene_id_to_pred)} 个预测文件")
    return scene_id_to_pred


def read_data_info(data_info_path):
    """
    读取 data_info 文件，获取 scene_id 对应的真值路径

    Args:
        data_info_path (str): data_info 文件路径

    Returns:
        dict: scene_id 到真值路径的映射
    """
    data_info_path = Path(data_info_path)

    if not data_info_path.exists():
        print(f"错误: data_info 文件 {data_info_path} 不存在")
        return {}

    try:
        df = pd.read_csv(data_info_path)
        # 如果有 scene_id 列，直接使用
        scene_id_to_gt = {}
        for _, row in df.iterrows():
            scene_num = row["scene_id"]
            scene_id = f"scene_{scene_num:05d}"  # 完整的 scene_id 格式
            gt_path = f"{row['path']}\\Annotations\\floorplan.dxf"
            scene_id_to_gt[scene_id] = gt_path

        print(f"从 {data_info_path} 中读取到 {len(scene_id_to_gt)} 条数据信息")
        return scene_id_to_gt
    except Exception as e:
        print(f"读取 {data_info_path} 时出错: {e}")
        return {}


def generate_eval_config(val_scene_ids, pred_files, gt_files, output_path):
    """
    生成评测配置文件

    Args:
        val_scene_ids (list): 验证数据的 scene_id 列表
        pred_files (dict): scene_id 到预测文件路径的映射
        gt_files (dict): scene_id 到真值路径的映射
        output_path (str): 输出配置文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_lines = []
    matched_count = 0

    # 遍历所有验证数据的 scene_id
    for scene_id in val_scene_ids:
        # 确保使用完整的 scene_id 格式 (scene_XXXXX)
        if not scene_id.startswith("scene_"):
            full_scene_id = f"scene_{scene_id}"
        else:
            full_scene_id = scene_id

        parts = full_scene_id.split("_")
        full_scene_id = "_".join(parts[:2])

        # 检查是否有对应的预测文件和真值文件
        if full_scene_id in pred_files and full_scene_id in gt_files:
            gt_path = gt_files[full_scene_id]
            pred_path = pred_files[full_scene_id]
            config_lines.append(f"{gt_path},{pred_path}")
            matched_count += 1
        else:
            missing = []
            if full_scene_id not in pred_files:
                missing.append("预测文件")
            if full_scene_id not in gt_files:
                missing.append("真值文件")
            print(f"警告: scene_id {full_scene_id} 缺少 {', '.join(missing)}")

    # 写入配置文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in config_lines:
            f.write(line + "\n")

    print(f"已生成配置文件: {output_path}")
    print(f"匹配的数据对数: {matched_count}/{len(val_scene_ids)}")


def parse_args():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="生成评测配置文件，用于评测程序")

    parser.add_argument(
        "--data_root", type=str, required=True, help="数据根目录路径，包含 val.csv 文件"
    )

    parser.add_argument(
        "--pred_path", type=str, required=True, help="预测文件路径，包含 dxf 文件"
    )

    parser.add_argument(
        "--data_info",
        type=str,
        required=True,
        help="data_info 文件路径，包含真值路径信息",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eval_config.txt",
        help="输出配置文件路径 (默认: eval_config.txt)",
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()

    print("开始生成评测配置文件...")
    print(f"数据根目录: {args.data_root}")
    print(f"预测文件路径: {args.pred_path}")
    print(f"data_info 文件: {args.data_info}")
    print(f"输出配置文件: {args.output}")
    print("-" * 50)

    # 1. 读取 val.csv 文件，获取验证数据对应的 scene_id
    val_scene_ids = read_val_csv(args.data_root)
    if not val_scene_ids:
        print("错误: 未找到验证数据，程序退出")
        return 1

    # 2. 获取用户指定预测路径下的 dxf 文件列表
    pred_files = get_pred_dxf_files(args.pred_path)
    if not pred_files:
        print("错误: 未找到预测文件，程序退出")
        return 1

    # 3. 读取 data_info 文件，获取 scene_id 对应的真值路径
    gt_files = read_data_info(args.data_info)
    if not gt_files:
        print("错误: 未读取到数据信息，程序退出")
        return 1

    # 4. 生成配置文件
    generate_eval_config(val_scene_ids, pred_files, gt_files, args.output)

    print("评测配置文件生成完成!")
    return 0


if __name__ == "__main__":
    exit(main())
