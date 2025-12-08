import argparse
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from spatiallm.layout.layout import Layout

random.seed(42)

prompt = (
    "<point_cloud>Detect walls, doors, windows. The reference code is as followed: \n\
@dataclass\n\
class Room:\n\
    id: int\n\
    wall_ids: List[str]\n\
    type: str\n\
@dataclass\n\
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


def generate_split_csv(
    data_root,
    split_file=None,
    train_ratio=0.8,
    pcd_dir_name="pcd",
    layout_dir_name="layout",
):
    """
    Generate CSV files containing the correspondence between point cloud and layout files.

    CSV format example:
    id,pcd,layout
    scene_00001,pcd/scene_00001.ply,layout/scene_00001.txt

    Args:
        data_root (str or Path): Root directory of the data
        split_file (str): Name of the generated split file
        train_ratio (float): Ratio of training data when splitting into train/val
        pcd_dir_name (str): Name of the point cloud directory
        layout_dir_name (str): Name of the layout directory
    """
    data_root = Path(data_root)
    pcd_dir = data_root / pcd_dir_name
    layout_dir = data_root / layout_dir_name

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
        # raise ValueError(error_msg)
        print(f"Warning: {error_msg}")

        common_stems = pcd_stems & layout_stems
        pcd_list = [f for f in pcd_list if f.stem in common_stems]

    if split_file is not None:
        split_file_path = data_root.joinpath(split_file)
        with split_file_path.open("w") as f:
            f.write("id,pcd,layout\n")
            for pcd_file in pcd_list:
                layout_file = layout_dir / f"{pcd_file.stem}.txt"
                f.write(
                    f"{pcd_file.stem},{pcd_file.relative_to(data_root)},{layout_file.relative_to(data_root)}\n"
                )
        print(f"Generated split file: {split_file_path}")
    else:
        # split into train and val
        train_size = int(len(pcd_list) * train_ratio)
        random.shuffle(pcd_list)
        train_list = sorted(pcd_list[:train_size])
        val_list = sorted(pcd_list[train_size:])

        for m in ["train", "val"]:
            split_file_path = data_root.joinpath(f"{m}.csv")
            with split_file_path.open("w") as f:
                f.write("id,pcd,layout\n")
                current_list = train_list if m == "train" else val_list
                for pcd_file in current_list:
                    layout_file = layout_dir / f"{pcd_file.stem}.txt"
                    f.write(
                        f"{pcd_file.stem},{pcd_file.relative_to(data_root)},{layout_file.relative_to(data_root)}\n"
                    )
            print(f"Generated split file: {split_file_path}")


def generate_train_csv(
    data_root,
    subfolders=None,
    val_csv_file="val.csv",
    pcd_dir_name="pcd",
    layout_dir_name="layout",
):
    """
    Generate training CSV file by excluding validation samples from all available data.

    Args:
        data_root (str or Path): Root directory of the data
        subfolders (list): List of subfolders to process. If None, search in data_root directly.
        val_csv_file (str): Name of the validation CSV file to exclude from training data
        pcd_dir_name (str): Name of the point cloud directory
        layout_dir_name (str): Name of the layout directory
    """
    data_root = Path(data_root)

    pcd_list = []
    layout_list = []

    if subfolders is None:
        pcd_dir = data_root / pcd_dir_name
        layout_dir = data_root / layout_dir_name

        # 检查目录是否存在
        if not pcd_dir.exists():
            raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")
        if not layout_dir.exists():
            raise FileNotFoundError(f"Layout directory not found: {layout_dir}")

        pcd_list.extend(sorted([f for f in pcd_dir.glob("*.ply")]))
        layout_list.extend(sorted([f for f in layout_dir.glob("*.txt")]))
    else:
        for subfolder in subfolders:
            pcd_dir = data_root.joinpath(subfolder, pcd_dir_name)
            layout_dir = data_root.joinpath(subfolder, layout_dir_name)

            # 检查目录是否存在
            if not pcd_dir.exists():
                raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")
            if not layout_dir.exists():
                raise FileNotFoundError(f"Layout directory not found: {layout_dir}")

            pcd_list.extend(sorted([f for f in pcd_dir.glob("*.ply")]))
            layout_list.extend(sorted([f for f in layout_dir.glob("*.txt")]))

    pcd_list = sorted(pcd_list)
    layout_list = sorted(layout_list)

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
        # raise ValueError(error_msg)
        print(f"Warning: {error_msg}")

        common_stems = pcd_stems & layout_stems
        pcd_list = [f for f in pcd_list if f.stem in common_stems]

    val_file_path = data_root.joinpath(val_csv_file)
    df = pd.read_csv(val_file_path)

    val_stems = df["id"].to_list()
    train_pcds = [s for s in pcd_list if s.stem not in val_stems]

    train_csv_path = data_root.joinpath("train.csv")
    with train_csv_path.open("w") as f:
        f.write("id,pcd,layout\n")
        for pcd_file in train_pcds:
            # layout_file = layout_dir / f"{pcd_file.stem}.txt"
            layout_file = pcd_file.parents[1] / layout_dir_name / f"{pcd_file.stem}.txt"
            f.write(
                f"{pcd_file.stem},{pcd_file.relative_to(data_root)},{layout_file.relative_to(data_root)}\n"
            )
    print(f"Generated split file: {train_csv_path}")


def generate_train_json(data_root, split_file="train.csv", dataset_name="HC3D"):
    """
    Generate training JSON dataset based on the split file.

    Args:
        data_root (str or Path): Root directory of the data
        split_file (str): Name of the split file
        dataset_name (str): Name of the dataset
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

    # 将数据写入JSON文件
    output_file = data_root / f"{dataset_name}_{mode}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Generated JSON file: {output_file}")
    print(f"Total samples: {len(data)}")


def generate_meta(data_root, dataset_name="HC3D", output_file="dataset_info.json"):
    """
    Generate metadata for the dataset.

    Creates a metadata JSON file that describes the dataset structure and formatting.
    The metadata includes information about training and validation datasets.

    Args:
        data_root (str or Path): Root directory of the data
        dataset_name (str): Name of the dataset

    Returns:
        None
    """
    data_root = Path(data_root)

    # Check for existing JSON files to determine which splits exist
    train_json = data_root / f"{dataset_name}_train.json"
    val_json = data_root / f"{dataset_name}_val.json"

    metadata = {}

    # Add training dataset metadata if file exists
    if train_json.exists():
        metadata[f"{dataset_name}_train"] = {
            "file_name": f"{dataset_name}_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
        }

    # Add validation dataset metadata if file exists
    if val_json.exists():
        metadata[f"{dataset_name}_val"] = {
            "file_name": f"{dataset_name}_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "point_clouds": "point_clouds"},
        }

    # Write metadata to JSON file
    output_file = data_root / output_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Generated metadata file: {output_file}")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate HC3D dataset from point cloud and layout files"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/stru3d_clipped",
        help="Root directory of the data (default: data/stru3d_clipped)",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stru3d",
        help="Name of the dataset (default: stru3d)",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.97,
        help="Ratio of training data when splitting into train/val (default: 0.97)",
    )

    parser.add_argument(
        "--pcd_dir",
        type=str,
        default="pcd",
        help="Name of the point cloud directory (default: pcd)",
    )

    parser.add_argument(
        "--layout_dir",
        type=str,
        default="layout",
        help="Name of the layout directory (default: layout)",
    )

    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=None,
        help="List of subfolders to process (default: None, search in data_root directly)",
    )

    parser.add_argument(
        "--skip_json",
        action="store_true",
        help="Skip JSON generation, only create CSV splits",
    )

    parser.add_argument(
        "--skip_meta", action="store_true", help="Skip metadata generation"
    )

    return parser.parse_args()


def main():
    """
    Main function to generate dataset splits and JSON files for training.

    This function orchestrates the dataset generation process by:
    1. Checking if val.csv exists and generating appropriate splits
    2. Optionally generating JSON files for training
    3. Optionally generating metadata file

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    data_root = Path(args.data_root)
    dataset_name = args.dataset_name
    train_ratio = args.train_ratio
    pcd_dir_name = args.pcd_dir
    layout_dir_name = args.layout_dir
    subfolders = args.subfolders
    skip_json = args.skip_json
    skip_meta = args.skip_meta

    try:
        # Step 1: Generate train/validation split CSV files
        print("Step 1: Generating train/validation split CSV files...")

        # Check if val.csv already exists
        val_csv_path = data_root / "val.csv"
        if val_csv_path.exists():
            print(f"Found existing validation file: {val_csv_path}")
            print("Generating training CSV based on existing validation split...")
            generate_train_csv(
                data_root,
                subfolders=subfolders,
                val_csv_file="val.csv",
                pcd_dir_name=pcd_dir_name,
                layout_dir_name=layout_dir_name,
            )
        else:
            print("No validation file found. Creating train/validation split...")
            generate_split_csv(
                data_root,
                train_ratio=train_ratio,
                pcd_dir_name=pcd_dir_name,
                layout_dir_name=layout_dir_name,
            )

        if not skip_json:
            # Step 2: Generate training JSON dataset
            print("Step 2: Generating training JSON dataset...")
            generate_train_json(
                data_root, split_file="train.csv", dataset_name=dataset_name
            )

            # Step 3: Generate validation JSON dataset
            print("Step 3: Generating validation JSON dataset...")
            generate_train_json(
                data_root, split_file="val.csv", dataset_name=dataset_name
            )
        else:
            print("Skipping JSON generation as requested")

        if not skip_meta:
            # Step 4: Generate metadata file
            print("Step 4: Generating metadata file...")
            generate_meta(data_root, dataset_name=dataset_name)
        else:
            print("Skipping metadata generation as requested")

        print("Dataset generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
