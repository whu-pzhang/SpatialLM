import numpy as np
from typing import List, Dict, Any
from copy import deepcopy
from spatiallm_hf.constants import NORMALIZATION_PRESET


class BaseEntity:
    def translate(self, translation: np.ndarray):
        raise NotImplementedError

    def rotate(self, angle: float):
        raise NotImplementedError

    def scale(self, scale: float):
        raise NotImplementedError

    def normalize_and_discretize(self, num_bins: int):
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class ZRangeEntity(BaseEntity):
    def __init__(self, data: Dict[str, Any]):
        self.vertices = np.array(data.get("vertices", []), dtype=np.float32)
        self.z_start = float(data.get("z_start", 0.0))
        self.z_end = float(data.get("z_end", 0.0))
        # Keep other fields?
        self.extra_data = {
            k: v for k, v in data.items() if k not in ["vertices", "z_start", "z_end"]
        }

    def translate(self, translation: np.ndarray):
        # translation is [tx, ty, tz]
        if len(self.vertices) > 0:
            self.vertices += translation[:2]
        self.z_start += translation[2]
        self.z_end += translation[2]

    def rotate(self, angle: float):
        # Rotate around Z axis
        if len(self.vertices) > 0:
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]])
            self.vertices = self.vertices @ rot_mat.T

    def scale(self, scale: float):
        if len(self.vertices) > 0:
            self.vertices *= scale
        self.z_start *= scale
        self.z_end *= scale

    def normalize_and_discretize(self, num_bins: int):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_factor = num_bins / (world_max - world_min)

        if len(self.vertices) > 0:
            self.vertices = (self.vertices - world_min) * scale_factor
            self.vertices = np.clip(self.vertices, 0, num_bins - 1).astype(int)

        self.z_start = (self.z_start - world_min) * scale_factor
        self.z_end = (self.z_end - world_min) * scale_factor

        self.z_start = np.clip(int(self.z_start), 0, num_bins - 1)
        self.z_end = np.clip(int(self.z_end), 0, num_bins - 1)

    def to_dict(self) -> Dict[str, Any]:
        d = deepcopy(self.extra_data)
        if len(self.vertices) > 0:
            d["vertices"] = self.vertices.tolist()
        else:
            d["vertices"] = []
        d["z_start"] = self.z_start
        d["z_end"] = self.z_end
        return d


class Door(ZRangeEntity):
    pass


class Window(ZRangeEntity):
    pass


class Room(BaseEntity):
    def __init__(self, data: Dict[str, Any]):
        self.vertices = np.array(data.get("vertices", []), dtype=np.float32)

        self.doors = [Door(d) for d in data.get("doors", [])]
        self.windows = [Window(w) for w in data.get("windows", [])]

        self.extra_data = {
            k: v for k, v in data.items() if k not in ["vertices", "doors", "windows"]
        }

    def translate(self, translation: np.ndarray):
        if len(self.vertices) > 0:
            self.vertices += translation[:2]
        for door in self.doors:
            door.translate(translation)
        for window in self.windows:
            window.translate(translation)

    def rotate(self, angle: float):
        if len(self.vertices) > 0:
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]])
            self.vertices = self.vertices @ rot_mat.T
        for door in self.doors:
            door.rotate(angle)
        for window in self.windows:
            window.rotate(angle)

    def scale(self, scale: float):
        if len(self.vertices) > 0:
            self.vertices *= scale
        for door in self.doors:
            door.scale(scale)
        for window in self.windows:
            window.scale(scale)

    def normalize_and_discretize(self, num_bins: int):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_factor = num_bins / (world_max - world_min)

        if len(self.vertices) > 0:
            self.vertices = (self.vertices - world_min) * scale_factor
            self.vertices = np.clip(self.vertices, 0, num_bins - 1).astype(int)

        for door in self.doors:
            door.normalize_and_discretize(num_bins)
        for window in self.windows:
            window.normalize_and_discretize(num_bins)

    def to_dict(self) -> Dict[str, Any]:
        d = deepcopy(self.extra_data)
        if len(self.vertices) > 0:
            d["vertices"] = self.vertices.tolist()
        else:
            d["vertices"] = []
        d["doors"] = [door.to_dict() for door in self.doors]
        d["windows"] = [window.to_dict() for window in self.windows]
        return d
