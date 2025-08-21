import json
import os

import pandas as pd
from tqdm import tqdm

from spatiallm.layout.layout import Layout

prompt = "<point_cloud>Detect walls, doors, windows. The reference code is as followed: @dataclass\nclass Wall:\n    ax: int\n    ay: int\n    az: int\n    bx: int\n    by: int\n    bz: int\n    height: int\n    thickness: int\n\n@dataclass\nclass Door:\n    wall_id: str\n    position_x: int\n    position_y: int\n    position_z: int\n    width: int\n    height: int\n\n@dataclass\nclass Window:\n    wall_id: str\n    position_x: int\n    position_y: int\n    position_z: int\n    width: int\n    height: int\n\n@dataclass\nclass Bbox:\n    class: str\n    position_x: int\n    position_y: int\n    position_z: int\n    angle_z: int\n    scale_x: int\n    scale_y: int\n    scale_z: int"


def main():
    data_root = "data/structured3d-spatiallm"
    for mode in ["train", "test"]:
        df = pd.read_csv(os.path.join(data_root, f"{mode}.csv"))
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ply_path = row["pcd"]
            layout_path = os.path.join(data_root, row["layout"])

            with open(layout_path) as f:
                layout_content = f.read()
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

        with open(os.path.join(data_root, f"structured3d_{mode}.json"), "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
