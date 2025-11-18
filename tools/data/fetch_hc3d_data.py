import argparse
from pathlib import Path

import pandas as pd
from natsort import natsorted


def check_file_integrity(file_path: Path) -> bool:
    """检查数据文件完整性

    验证以下必需文件是否存在：
    - Annotations/floorplan.json
    - Annotations/floorplan.dxf
    """
    required_files = [
        file_path / "Annotations" / "floorplan.json",
        file_path / "Annotations" / "floorplan.dxf",
    ]

    return all(req_file.exists() for req_file in required_files)


def save_data_info_to_csv(data_infos, csv_path: Path):
    """将数据信息保存到CSV文件"""
    df = pd.DataFrame(data_infos)
    df.to_csv(csv_path, index=False)


def add_scene_id(data_infos, start_idx=1):
    """为数据信息添加场景ID"""
    data_infos = natsorted(data_infos, key=lambda x: x["data_name"])
    for idx, info in enumerate(data_infos, start=start_idx):
        info["scene_id"] = f"{idx:05d}"
    return data_infos


def read_processed_data(data_root: Path) -> dict:
    """从 data_root 目录中自动识别并读取所有已处理的数据记录"""
    processed_data = {}

    # 查找所有 data_info_for_llm 开头的 txt 文件
    pattern = "data_info_for_llm_*.txt"
    processed_files = list(data_root.glob(pattern))

    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            file_data = {d["data_name"]: d["scene_id"] for _, d in df.iterrows()}
            processed_data.update(file_data)
            print(f"已读取处理记录文件: {file_path.name}")
        except Exception as e:
            print(f"读取文件 {file_path.name} 时出错: {e}")

    return processed_data


def fetch_data_info_incremental(
    data_root: Path, device_folders: list[str]
) -> list[dict]:
    """增量式获取数据信息"""
    processed_data = read_processed_data(data_root)
    data_infos = []

    for device in device_folders:
        device_path = data_root / device
        if not device_path.exists():
            print(f"Device folder {device} does not exist.")
            continue

        for subject_folder in device_path.iterdir():
            if not subject_folder.is_dir():
                continue

            if subject_folder.name in processed_data:
                print(f"Skipping already processed data: {subject_folder.name}")
                continue

            if check_file_integrity(subject_folder):
                place, house_type, furnish_type = subject_folder.name.split("_")[:3]
                data_infos.append(
                    {
                        "data_name": subject_folder.name,
                        "place": place,
                        "house_type": house_type,
                        "device": device,
                        "furnish_type": furnish_type,
                        "path": str(subject_folder),
                    }
                )

    # scene_id
    start_idx = 1
    if processed_data:
        start_idx += len(processed_data)

    return add_scene_id(data_infos, start_idx=start_idx)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="获取和处理数据信息")
    parser.add_argument(
        "--scanner_data_root",
        type=str,
        default="/mnt/DATA108/01_3D-FAVP/handheld_scanner_Data",
        help="手持扫描仪数据根目录路径",
    )
    parser.add_argument(
        "--scanner_devices",
        nargs="+",
        default=["RS10", "S20"],
        help="扫描仪设备类型列表",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="data_info_for_llm",
        help="输出文件名前缀（将自动添加时间戳）",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    scanner_data_root = Path(args.scanner_data_root)
    scanner_devices = args.scanner_devices

    data_infos = fetch_data_info_incremental(scanner_data_root, scanner_devices)
    print(f"找到 {len(data_infos)} 条新数据")

    if data_infos:
        # 添加时间戳到文件名
        timestamp = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        output_filename = f"{args.output_prefix}_{timestamp}.txt"
        output_path = scanner_data_root / output_filename
        save_data_info_to_csv(data_infos, output_path)
        print(f"数据信息已保存到 {output_path}")


if __name__ == "__main__":
    main()
