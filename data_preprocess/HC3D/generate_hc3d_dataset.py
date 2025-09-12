import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from spatiallm.layout.layout import Layout

prompt = (
    "<point_cloud>Detect walls, doors, windows. The reference code is as followed: @dataclass\n\
class Wall:\n\
    ax: int\n\
    ay: int\n\
    az: int\n\
    bx: int\n\
    by: int\n\
    bz: int\n\
    height: int\n\
    thickness: int\n\n\
@dataclass\n\
class Door:\n\
    wall_id: str\n\
    position_x: int\n\
    position_y: int\n\
    position_z: int\n\
    width: int\n\
    height: int\n\n\
@dataclass\n\
class Window:\n\
    wall_id: str\n\
    position_x: int\n\
    position_y: int\n\
    position_z: int\n\
    width: int\n\
    height: int\n\n\
@dataclass\n\
class Bbox:\n\
    class: str\n\
    position_x: int\n\
    position_y: int\n\
    position_z: int\n\
    angle_z: int\n\
    scale_x: int\n\
    scale_y: int\n\
    scale_z: int"
)


def generate_split_csv(data_root, split_file):
    """
    生成包含点云和布局文件对应关系的CSV文件

    CSV格式示例:
    id,pcd,layout
    scene_00000,pcd/scene_00000.ply,layout/scene_00000.txt

    Args:
        data_root (str or Path): 数据根目录
        split_file (str): 生成的分割文件名
    """
    data_root = Path(data_root)
    pcd_dir = data_root / "pcd"
    layout_dir = data_root / "layout"

    # 检查目录是否存在
    if not pcd_dir.exists():
        raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_dir}")

    pcd_list = sorted([f for f in pcd_dir.glob("*.ply")])
    layout_list = sorted([f for f in layout_dir.glob("*.txt")])

    # 检查点云文件和布局文件是否匹配
    pcd_stems = set([f.stem for f in pcd_list])
    layout_stems = set([f.stem for f in layout_list])

    if pcd_stems != layout_stems:
        missing_in_pcd = layout_stems - pcd_stems
        missing_in_layout = pcd_stems - layout_stems
        error_msg = "PCD and Layout files mismatch."
        if missing_in_pcd:
            error_msg += f" Missing PCD files for layouts: {missing_in_pcd}"
        if missing_in_layout:
            error_msg += f" Missing layout files for PCDs: {missing_in_layout}"
        raise ValueError(error_msg)

    split_file_path = data_root.joinpath(split_file)
    with split_file_path.open("w") as f:
        f.write("id,pcd,layout\n")
        for pcd_file in pcd_list:
            layout_file = layout_dir / f"{pcd_file.stem}.txt"
            f.write(
                f"{pcd_file.stem},{pcd_file.relative_to(data_root)},{layout_file.relative_to(data_root)}\n"
            )
    print(f"Generated split file: {split_file_path}")


def generate_train_json(data_root, split_file="train.csv", dataset_name="HC3D"):
    """
    根据分割文件生成训练用JSON数据集

    Args:
        data_root (str or Path): 数据根目录
        split_file (str): 分割文件名
        dataset_name (str): 数据集名称
    """
    data_root = Path(data_root)
    mode = Path(split_file).stem

    split_file_path = data_root.joinpath(split_file)
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    df = pd.read_csv(split_file_path)
    data = []

    print(f"Processing {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ply_path = row["pcd"]
        layout_path = data_root / row["layout"]

        # 检查文件是否存在
        if not layout_path.exists():
            print(f"Warning: Layout file not found: {layout_path}")
            continue

        try:
            with open(layout_path) as f:
                layout_content = f.read()

            # 解析布局内容并转换为语言字符串
            layout = Layout(layout_content)
            layout_str = layout.to_language_string()

            data.append(
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt,
                        },
                        {
                            "from": "gpt",
                            "value": f"<|layout_s|>{layout_str}<|layout_e|>",
                        },
                    ],
                    "point_clouds": [ply_path],
                }
            )
        except Exception as e:
            print(f"Error processing {layout_path}: {e}")
            continue

    # 保存JSON文件
    output_file = data_root.joinpath(f"{dataset_name}_{mode}.json")
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated JSON dataset: {output_file} with {len(data)} samples")


def main():
    data_root = "data/HC3D/processed"
    split_file = "train.csv"
    dataset_name = "HC3D"

    try:
        generate_split_csv(data_root, split_file=split_file)
        generate_train_json(data_root, split_file=split_file, dataset_name=dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
