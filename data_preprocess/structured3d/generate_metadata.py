# Convert structured3d-spatiallm dataset's split.csv to
# train.csv and test.csv
# id,pcd,layout
import os

import pandas as pd


def main(args):
    data_root = args.data_root
    pcd_dir = args.pcd_dir
    layout_dir = args.layout_dir

    split_file = os.path.join(data_root, "split.csv")
    split_df = pd.read_csv(split_file)

    for mode in ["train", "test"]:
        # iterate over rows
        train_df = split_df[split_df["split"] == mode].copy()

        train_df["pcd"] = train_df["id"].apply(
            lambda x: os.path.join(pcd_dir, f"{x}.ply")
        )
        train_df["layout"] = train_df["id"].apply(
            lambda x: os.path.join(layout_dir, f"{x}.txt")
        )
        # remove split column
        train_df = train_df.drop(columns=["split"])
        # save train.csv
        train_df.to_csv(os.path.join(data_root, f"{mode}.csv"), index=False)
        print(f"Saved {mode}.csv with {len(train_df)} entries.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate metadata for structured3d dataset"
    )
    parser.add_argument(
        "data_root",
        type=str,
        default="data/structured3d-spatiallm",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        default="pcd",
        help="Directory containing point cloud files",
    )
    parser.add_argument(
        "--layout_dir",
        type=str,
        default="layout",
        help="Directory containing layout files",
    )

    args = parser.parse_args()

    main(args)
