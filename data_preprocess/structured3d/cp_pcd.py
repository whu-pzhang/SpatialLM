from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil


data_root = Path("data/structured3d-spatiallm")
dst_dir = data_root / "pcd_test"
dst_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(data_root / "test.csv")

for _, row in tqdm(df.iterrows(), total=len(df)):
    pcd_path = data_root / row["pcd"]
    # copy pcd to dst_dir
    shutil.copy(pcd_path, dst_dir / pcd_path.name)


