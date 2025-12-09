import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from spatiallm_hf.constants import NORMALIZATION_PRESET
from spatiallm_hf.pcd.pcd_loader import load_o3d_pcd, get_points_and_colors
from spatiallm_hf.pcd.transform import Compose
from spatiallm_hf.layout.layout import Layout

logger = logging.getLogger(__name__)


class SpatialLMDataset(Dataset):
    """SpatialLM 数据集（点云 + 布局）

    读取 JSONL 标注，加载并增广点云，同步布局几何，进行体素采样并输出张量化数据。

    - 支持随机旋转/缩放与颜色增强
    - 支持分布式训练数据分片（可选）
    - 输出包含 `grid_coord/coord/color/rooms/pcd_path`
    """

    def __init__(
        self,
        jsonl_file: str,
        data_root: str,
        num_bins: int = 1280,
        do_augmentation: bool = False,
        random_rotation: bool = False,
        random_scale: bool = False,
        max_num_samples: Optional[int] = None,
        shuffle: bool = False,
        shard_by_rank: bool = False,
    ) -> None:
        """初始化数据集

        参数：
        - `jsonl_file`: JSONL 标注文件路径
        - `data_root`: PCD 文件根目录（JSONL 使用相对路径）
        - `num_bins`: 世界坐标离散桶数，用于网格采样与布局离散
        - `do_augmentation`: 是否进行颜色/噪声增强
        - `random_rotation`: 是否进行随机 Z 轴旋转（否则使用四象限离散角）
        - `random_scale`: 是否进行随机缩放
        - `max_num_samples`: 限制最大样本数
        - `shuffle`: 读取后打乱样本顺序
        - `shard_by_rank`: 是否按分布式 rank 对样本进行切片
        """
        self.data_root = Path(data_root)
        self.items: List[Dict[str, Any]] = []
        jsonl_path = Path(jsonl_file)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON line: {e}")
                        raise

        if shuffle:
            random.shuffle(self.items)

        if max_num_samples is not None and max_num_samples > 0:
            self.items = self.items[:max_num_samples]

        # 分布式切片（可选）
        if shard_by_rank:
            world_size = 1
            rank = 0
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                else:
                    world_size = int(os.environ.get("WORLD_SIZE", "1"))
                    rank = int(os.environ.get("RANK", "0"))
            except Exception as e:
                logger.warning(f"Distributed context not available: {e}")

            if world_size > 1:
                total = len(self.items)
                per_rank = total // world_size
                start = rank * per_rank
                end = (rank + 1) * per_rank if rank < world_size - 1 else total
                logger.info(
                    f"Sharding dataset: world_size={world_size}, rank={rank}, range=({start}, {end})"
                )
                self.items = self.items[start:end]

        self.num_bins = num_bins
        self.do_augmentation = do_augmentation
        self.random_rotation = random_rotation
        self.random_scale = random_scale

        # Basic transform from mm_plugin
        global_extent = NORMALIZATION_PRESET["world"]
        self.grid_size = (global_extent[1] - global_extent[0]) / self.num_bins
        self.basic_transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=self.grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=self.num_bins,
                ),
            ]
        )

        # Augmentation pipeline from mm_plugin.py
        if self.do_augmentation:
            self.augmentation = Compose(
                [
                    dict(type="RandomColorGrayScale", p=0.05),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.75, ratio=0.1),
                    dict(type="ChromaticJitter", p=0.8, std=0.05),
                    dict(
                        type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2
                    ),
                    dict(type="RandomColorDrop", p=0.1, color_augment=0.0),
                    dict(type="RandomJitter", sigma=0.025, clip=0.05, ratio=0.8, p=0.9),
                    dict(type="RandomJitter", sigma=0.2, clip=0.2, ratio=0.05, p=0.85),
                    dict(type="RandomJitter", sigma=0.4, clip=1.0, ratio=0.001, p=0.75),
                    dict(type="RandomJitter", sigma=0.5, clip=4.0, ratio=0.0005, p=0.7),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                        p=[0.85, 0.5],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        pcd_rel_path = item["pcd_path"]
        pcd_path = self.data_root / pcd_rel_path

        if not pcd_path.exists():
            logger.error(f"PCD file not found: {pcd_path}")
            raise FileNotFoundError(f"PCD file not found: {pcd_path}")

        # 加载点云
        pcd = load_o3d_pcd(str(pcd_path))
        points, colors = get_points_and_colors(pcd)

        # 颜色/噪声增广
        if self.do_augmentation:
            data_aug = {"coord": points, "color": colors}
            data_aug = self.augmentation(data_aug)
            points = data_aug["coord"]
            colors = data_aug["color"]

        # 几何增广（旋转/缩放）
        if self.random_rotation:
            angle_z = np.random.random() * 2 * np.pi
        else:
            angle_z = np.random.choice(np.array([0, 0.5, 1.0, 1.5]) * np.pi)

        if self.random_scale:
            scaling = np.random.uniform(0.75, 1.25)
        else:
            scaling = 1.0

        # 应用几何变换到点云
        # 1) 以包围盒中心居中
        min_bound_raw = points.min(axis=0)
        max_bound_raw = points.max(axis=0)
        center_pt = (min_bound_raw + max_bound_raw) / 2

        rotmat = R.from_rotvec(np.array([0, 0, angle_z])).as_matrix()

        scaled_points = (points - center_pt) * scaling
        transformed_points = (rotmat @ scaled_points.T).T + center_pt

        # 计算新的 min_bound 用于正移与布局同步
        min_bound = transformed_points.min(axis=0)

        # Prepare transformations dict for layout sync
        # transformations = {
        #     "angle_z": angle_z,
        #     "center_pt": center_pt,
        #     "scaling": scaling,
        #     "min_bound": min_bound,
        # }

        # 按 min_bound 正移，与布局同步
        transformed_points = transformed_points - min_bound

        # 布局同步与离散
        layout = Layout(item["rooms"])
        layout.translate(-center_pt)
        layout.scale(scaling)
        layout.rotate(angle_z)
        layout.translate(center_pt)
        layout.translate(-min_bound)
        layout.normalize_and_discretize(self.num_bins)
        rooms = layout.to_list()

        # 基础变换（体素采样、归一化颜色）
        data_dict = {
            "coord": transformed_points.astype(np.float32),
            "color": colors.astype(np.float32),
        }
        data_dict = self.basic_transform(data_dict)

        # 采样后保留全部点（`GridSample` 返回 `grid_coord/coord/color`）

        # 附加房间与路径信息
        data_dict["rooms"] = rooms
        data_dict["pcd_path"] = str(pcd_path)

        # 转为张量
        for k in ["grid_coord", "coord", "color"]:
            if k in data_dict and isinstance(data_dict[k], np.ndarray):
                data_dict[k] = torch.from_numpy(data_dict[k]).float()

        return data_dict
