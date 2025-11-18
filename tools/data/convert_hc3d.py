# -*- coding: utf-8 -*-
"""
一个统一的脚本，用于处理HC3D数据集，包含以下功能：
1. LAS to PLY: 根据floorplan.json中的boundingBox裁剪.las点云，并保存为.ply格式。
2. DXF to TXT: 将3D.dxf文件转换为SpatialLM所需的文本格式。

"""

import argparse
import json
import logging
import multiprocessing
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ezdxf
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
from ezdxf.math import OCS, Vec3
from tqdm import tqdm

# ===================================================================================
# Logging System
# ===================================================================================


def setup_logger(name, log_level=logging.INFO, log_file=None, console=True):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_file: 日志文件路径，如果为None则不写入文件
        console: 是否输出到控制台

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.handlers:
        logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """获取日志记录器"""
    if name is None:
        name = "preprocess_hc3d"
    return logging.getLogger(name)


# ===================================================================================
# Helper Functions & Data Classes (Shared or generic)
# ===================================================================================


def parse_csv(file_path):
    """Reads a CSV file into a pandas DataFrame."""
    logger = get_logger()
    logger.debug(f"Reading CSV file: {file_path}")
    return pd.read_csv(file_path)


@dataclass
class Entity:
    """Represents a geometric entity from a DXF layer."""

    type: str
    id: str
    points: List[tuple] = field(default_factory=list)
    layer: Optional[str] = None


# ===================================================================================
# DXF to TXT Conversion Module
# ===================================================================================


@dataclass
class Wall:
    ax: float
    ay: float
    az: float
    bx: float
    by: float
    bz: float
    height: float
    thickness: float
    id: Optional[int] = None
    room_type: Optional[str] = None
    normal: Optional[np.ndarray] = None

    def to_language_string(self):
        room_comment = f"  # {self.room_type}" if self.room_type else ""
        return f"wall_{self.id}=Wall({self.ax:.9g},{self.ay:.9g},{self.az},{self.bx:.9g},{self.by:.9g},{self.bz},{self.height:.9g},{int(self.thickness)}){room_comment}"


@dataclass
class Door:
    wall_id: int
    cx: float
    cy: float
    cz: float
    width: float
    height: float

    def to_language_string(self, door_id: int):
        return f"door_{door_id}=Door(wall_{self.wall_id},{self.cx:.9g},{self.cy:.9g},{self.cz:.9g},{self.width:.9g},{self.height:.9g})"


@dataclass
class Window:
    wall_id: int
    cx: float
    cy: float
    cz: float
    width: float
    height: float

    def to_language_string(self, window_id: int):
        return f"window_{window_id}=Window(wall_{self.wall_id},{self.cx:.9g},{self.cy:.9g},{self.cz:.9g},{self.width:.9g},{self.height:.9g})"


def get_points_from_lwpolyline(e):
    elev = float(getattr(e.dxf, "elevation", 0.0))
    extrusion = getattr(e.dxf, "extrusion", Vec3(0, 0, 1))
    ocs = OCS(extrusion)
    pts = [ocs.to_wcs((float(raw[0]), float(raw[1]), elev)) for raw in e.get_points()]
    return [(p.x, p.y, p.z) for p in pts]


def remove_redundant_points(points, tolerance=1e-1):
    if len(points) <= 2:
        return points
    points = np.array(points)
    points = remove_duplicate_points(points, tolerance)
    if len(points) <= 2:
        return [tuple(p) for p in points]
    points = remove_collinear_points(points, tolerance)
    return [tuple(p) for p in points]


def remove_duplicate_points(points, tolerance=1e-6):
    if len(points) <= 1:
        return points
    points = np.array(points)
    while len(points) > 2:
        if np.linalg.norm(points[-1] - points[0]) < tolerance:
            points = points[:-1]
        else:
            break
    unique_points = []
    for point in points:
        if not any(np.linalg.norm(point - up) < tolerance for up in unique_points):
            unique_points.append(point)
    return np.array(unique_points)


def remove_collinear_points(points, tolerance=1e-6):
    if len(points) <= 2:
        return points
    points = np.array(points)
    changed = True
    while changed and len(points) > 2:
        changed = False
        new_points = []
        for i in range(len(points)):
            prev_i = (i - 1) % len(points)
            next_i = (i + 1) % len(points)
            dist_to_line = point_to_line_distance(
                points[i], points[prev_i], points[next_i]
            )
            if dist_to_line > tolerance:
                new_points.append(points[i])
            else:
                changed = True
        if changed and len(new_points) > 2:
            points = np.array(new_points)
        else:
            break
    return points


def point_to_line_distance(point, line_start, line_end):
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq < 1e-12:
        return np.linalg.norm(point - line_start)
    t = max(0, min(1, np.dot(point - line_start, line_vec) / line_length_sq))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def parse_layer_name_for_3d(layer_name):
    name_lower = layer_name.lower()
    if name_lower.startswith("room"):
        return "room", name_lower
    if name_lower.startswith("door"):
        return "door", name_lower
    if "window" in name_lower:
        return "window", name_lower
    return None, None


def is_coplanar_and_center_in_wall(face, wall, tolerance=0.01):
    if abs(abs(np.dot(wall.normal, face["normal"])) - 1.0) > tolerance:
        return False, float("inf")

    wall_start_2d = np.array([wall.ax, wall.ay])
    wall_end_2d = np.array([wall.bx, wall.by])
    wall_vec_2d = wall_end_2d - wall_start_2d
    wall_length = np.linalg.norm(wall_vec_2d)
    if wall_length < tolerance:
        return False, float("inf")

    wall_dir_norm = wall_vec_2d / wall_length
    face_center_2d = np.array([face["center"][0], face["center"][1]])
    face_to_wall_start = face_center_2d - wall_start_2d

    projection = np.dot(face_to_wall_start, wall_dir_norm)
    if not (0 <= projection <= wall_length):
        return False, float("inf")

    distance = np.linalg.norm(face_to_wall_start - projection * wall_dir_norm)
    if distance > tolerance:
        return False, distance

    return True, distance


def extrude_entities_to_3d(entities, z_min, rel_z_min, rel_z_max, min_wall_width=0.005):
    """Extrudes 2D floorplan entities (rooms, doors, windows) to 3D objects."""
    walls, doors, windows = [], [], []
    wall_id_counter = 0

    face_height = rel_z_max - rel_z_min

    # Adjust rel_z_max to absolute Z coordinate
    abs_z_min = z_min + rel_z_min
    abs_z_max = z_min + rel_z_max
    z_min = np.array([0, 0, abs_z_min])

    # Pass 1: Extrude rooms to create walls
    for (etype, eid), entity_list in entities.items():
        if etype != "room":
            continue

        for entity in entity_list:
            points = entity.points
            if points and np.allclose(points[0], points[-1]):
                points = points[:-1]

            for i in range(len(points)):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % len(points)])

                # Convert to absolute Z coordinates
                p1 += z_min
                p2 += z_min

                width = np.linalg.norm(p2 - p1)
                if width < min_wall_width:
                    continue

                face_vec = p2[:2] - p1[:2]
                normal = np.cross(np.append(face_vec, 0), [0, 0, 1])
                normal /= np.linalg.norm(normal)

                walls.append(
                    Wall(
                        p1[0],
                        p1[1],
                        p1[2],
                        p2[0],
                        p2[1],
                        p2[2],
                        height=face_height,
                        thickness=0.0,
                        id=wall_id_counter,
                        room_type=f"room_{eid}",
                        normal=normal,
                    )
                )
                wall_id_counter += 1

    # Pass 2: Create openings (doors, windows)
    for (etype, eid), entity_list in entities.items():
        if etype not in ["door", "window"]:
            continue
        min_door_width = max(min_wall_width * 0.2, 0.01)

        for entity in entity_list:
            points = entity.points
            if points and np.allclose(points[0], points[-1]):
                points = points[:-1]

            for i in range(len(points)):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % len(points)])

                # Convert to absolute Z coordinates
                p1 += z_min
                p2 += z_min

                center = (p1 + p2) / 2
                width = np.linalg.norm(p2 - p1)
                if width < min_door_width:
                    continue

                face_vec = p2[:2] - p1[:2]
                normal = np.cross(np.append(face_vec, 0), [0, 0, 1])
                normal /= np.linalg.norm(normal)

                # Create a temporary face dict to use with find_matching_wall
                face = {
                    "center": np.array(
                        [center[0], center[1], abs_z_min + face_height / 2]
                    ),
                    "width": width,
                    "height": face_height,
                    "normal": normal,
                    "bottom_p1": p1,
                    "bottom_p2": p2,
                }

                best_wall, best_distance = find_matching_wall(face, walls)
                if best_wall:
                    target_list = doors if etype == "door" else windows
                    target_list.append(
                        globals()[etype.capitalize()](
                            best_wall.id,
                            face["center"][0],
                            face["center"][1],
                            face["center"][2],
                            face["width"],
                            face["height"],
                        )
                    )

    return walls, doors, windows


def infer_vertical_faces(entity_list, min_wall_width=0.005, min_wall_height=0.08):
    """Infer vertical faces from horizontal polygons. 3D.dxf format."""
    faces = []
    horizontal_polys = [
        ent
        for ent in entity_list
        if len(ent.points) > 2 and np.std(np.array(ent.points)[:, 2]) < 5e-3
    ]
    if len(horizontal_polys) < 2:
        return faces

    z_groups = defaultdict(list)
    for poly in horizontal_polys:
        avg_z = np.mean([p[2] for p in poly.points])
        z_groups[round(avg_z, 3)].append(poly)

    if len(z_groups) < 2:
        return faces

    sorted_zs = sorted(z_groups.keys())
    bottom_z = sorted_zs[0]

    top_z = sorted_zs[-1]
    height = abs(top_z - bottom_z)

    if height < min_wall_height:
        return faces

    for bottom_poly in z_groups[bottom_z]:
        bottom_points = bottom_poly.points
        if bottom_points and np.allclose(bottom_points[0], bottom_points[-1]):
            bottom_points = bottom_points[:-1]
        for i in range(len(bottom_points)):
            bp1, bp2 = (
                np.array(bottom_points[i]),
                np.array(bottom_points[(i + 1) % len(bottom_points)]),
            )
            width = np.linalg.norm(bp1 - bp2)
            if width < min_wall_width:
                continue

            center_3d = (bp1 + bp2) / 2
            center_3d[2] = bottom_z + height / 2
            face_vec = bp2[:2] - bp1[:2]
            face_normal = np.cross(np.append(face_vec, 0), [0, 0, 1])
            face_normal /= np.linalg.norm(face_normal)

            faces.append(
                {
                    "center": center_3d,
                    "width": width,
                    "height": height,
                    "normal": face_normal,
                    "bottom_p1": bp1,
                    "bottom_p2": bp2,
                }
            )
    return faces


def find_matching_wall(face, walls, tolerance=1e-2):
    best_wall, best_distance = None, float("inf")
    for wall in walls:
        is_matching, distance = is_coplanar_and_center_in_wall(face, wall, tolerance)
        if is_matching and distance < best_distance:
            best_wall, best_distance = wall, distance
    return best_wall, best_distance


def process_dxf_to_txt(src_path, dst_path, scene_name, args):
    """Processes a single DXF file and converts it to a TXT file."""
    logger = get_logger()
    logger.info(f"Processing DXF to TXT for: {scene_name}")

    dxf_filename = "floorplan.dxf" if args.extrude_from_floorplan else args.dxf_filename
    input_file = src_path / "Annotations" / dxf_filename
    if not input_file.is_file():
        logger.warning(f"DXF file not found at {input_file}. Skipping.")
        return False

    if args.extrude_from_floorplan and args.rel_z_max is None:
        logger.error(
            "--extrude_from_floorplan requires --max_height to be set. Skipping."
        )
        return False

    txt_filename = (
        f"{scene_name}_{args.voxel_size}.txt"
        if args.downsample
        else f"{scene_name}.txt"
    )
    output_file = dst_path.parent / "layout" / txt_filename

    try:
        logger.debug(f"Reading DXF file: {input_file}")
        doc = ezdxf.readfile(str(input_file))
        msp = doc.modelspace()
        logger.debug(f"Successfully loaded DXF file with {len(msp)} entities")
    except (IOError, ezdxf.DXFStructureError) as e:
        logger.error(f"Error reading DXF file {str(input_file)}: {e}")
        return False

    # read bounding box from floorplan.json
    json_path = src_path / "Annotations" / "floorplan.json"
    logger.debug(f"Reading bounding box from: {json_path}")
    bounding_box = read_floorplan_json(str(json_path))
    z_min = bounding_box[2]
    logger.debug(f"Bounding box: {bounding_box}")

    entities = defaultdict(list)
    if dxf_filename == "3D.dxf":
        for e in msp.query("LWPOLYLINE POLYLINE"):
            entity_type, entity_id = parse_layer_name_for_3d(e.dxf.layer)
            if not entity_type:
                continue

            if e.dxftype() == "LWPOLYLINE":
                points = get_points_from_lwpolyline(e)
            else:
                points = [v.dxf.location for v in e.vertices]

            if not args.no_point_cleanup:
                points = remove_redundant_points(points, args.point_tolerance)

            entities[(entity_type, entity_id)].append(
                Entity(type=entity_type, id=entity_id, points=points, layer=e.dxf.layer)
            )
    # TODO: 对2d floorplan.dxf, 其中的门和窗都分别放在相同的图层中，
    # 需要首先将不同的门窗实体区分开来
    elif dxf_filename == "floorplan.dxf":
        counters = defaultdict(int)
        for e in msp.query("LWPOLYLINE POLYLINE"):
            layer_names = e.dxf.layer.lower().split("_")
            entity_type = layer_names[0] if len(layer_names) == 1 else layer_names[1]
            counters[entity_type] += 1
            points = (
                get_points_from_lwpolyline(e)
                if e.dxftype() == "LWPOLYLINE"
                else [v.dxf.location for v in e.vertices]
            )
            if not args.no_point_cleanup:
                points = remove_redundant_points(points, args.point_tolerance)

            entities[(entity_type, counters[entity_type])].append(
                Entity(
                    type=entity_type,
                    id=counters[entity_type],
                    points=points,
                    layer=e.dxf.layer,
                )
            )
    else:
        print(f"  - Error: Unsupported DXF filename '{dxf_filename}'. Skipping.")
        return False

    walls, doors, windows = [], [], []

    if args.extrude_from_floorplan:
        logger.info("Extruding from 2D floorplan...")
        walls, doors, windows = extrude_entities_to_3d(
            entities, z_min, args.rel_z_min, args.rel_z_max, args.min_wall_width
        )
        logger.info(
            f"Generated {len(walls)} walls, {len(doors)} doors, {len(windows)} windows"
        )
    else:
        wall_id_counter = 0
        for (etype, eid), entity_list in entities.items():
            if etype != "room":
                continue
            faces = infer_vertical_faces(entity_list, args.min_wall_width, 0.01)
            for face in faces:
                p1, p2 = face["bottom_p1"], face["bottom_p2"]
                walls.append(
                    Wall(
                        p1[0],
                        p1[1],
                        p1[2],
                        p2[0],
                        p2[1],
                        p2[2],
                        face["height"],
                        0.0,
                        wall_id_counter,
                        f"room_{eid}",
                        face["normal"],
                    )
                )
                wall_id_counter += 1

        for (etype, eid), entity_list in entities.items():
            if etype not in ["door", "window"]:
                continue
            min_width = max(args.min_wall_width * 0.2, 0.01)
            faces = infer_vertical_faces(entity_list, min_width, 0.01)
            for face in faces:
                best_wall, _ = find_matching_wall(face, walls, args.tolerance)
                if best_wall:
                    target_list = doors if etype == "door" else windows
                    target_list.append(
                        globals()[etype.capitalize()](
                            best_wall.id,
                            face["center"][0],
                            face["center"][1],
                            face["center"][2],
                            face["width"],
                            face["height"],
                        )
                    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        walls.sort(key=lambda w: w.id)
        for w in walls:
            f.write(w.to_language_string() + "\n")

        doors.sort(key=lambda d: (d.cx, d.cy, d.cz))
        for i, d in enumerate(doors):
            f.write(d.to_language_string(i) + "\n")

        windows.sort(key=lambda w: (w.cx, w.cy, w.cz))
        for i, w in enumerate(windows):
            f.write(w.to_language_string(i) + "\n")

    logger.info(f"Successfully processed '{input_file.name}' -> '{output_file}'")
    return True


# ===================================================================================
# LAS to PLY Conversion Module
# ===================================================================================


def read_floorplan_json(json_path):
    """Reads a floorplan.json file and extracts the boundingBox."""
    logger = get_logger()
    logger.debug(f"Reading floorplan JSON from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frame = data["frames"][0]
    bounding_box = frame["boundingBox"]
    logger.debug(f"Extracted bounding box: {bounding_box}")
    return bounding_box


def load_las_file(las_path):
    """Loads a LAS file, extracting coordinates and colors."""
    logger = get_logger()
    logger.debug(f"Loading LAS file: {las_path}")
    las_file = laspy.read(las_path)
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    logger.debug(f"Loaded {len(points)} points from LAS file")
    colors = None
    try:
        if (
            hasattr(las_file, "red")
            and hasattr(las_file, "green")
            and hasattr(las_file, "blue")
        ):
            scale = np.array(
                [las_file.red.max(), las_file.green.max(), las_file.blue.max()]
            )
            if scale.any():  # ignore zero scale
                rgb = np.vstack((las_file.red, las_file.green, las_file.blue)).T
                logger.info(
                    f"RGB color information detected (converted from {rgb.dtype})."
                )
                colors = np.round(rgb / scale * 255.0).astype(np.uint8)
            else:
                logger.info(
                    "No RGB color information detected. Adding gray color (128, 128, 128)."
                )
                colors = np.full((len(points), 3), 128, dtype=np.uint8)
        else:
            logger.info(
                "No RGB color information detected. Adding gray color (128, 128, 128)."
            )
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Error reading RGB color information: {e}")
        logger.info("Adding gray color (128, 128, 128) as fallback.")
        colors = np.full((len(points), 3), 128, dtype=np.uint8)
    return points, colors


def crop_point_cloud(points, colors, bounding_box, rel_z_min=None, rel_z_max=None):
    """Crops a point cloud using a bounding box and an optional relative height limit."""
    x_min, y_min, z_min, x_max, y_max, z_max = bounding_box

    # Validate and normalize min/max heights
    if rel_z_min is not None and rel_z_max is not None:
        if rel_z_min > rel_z_max:
            print(
                f"  - Warning: rel_z_min ({rel_z_min}) > rel_z_max ({rel_z_max}). Swapping the values."
            )
            rel_z_min, rel_z_max = rel_z_max, rel_z_min

    # Compute absolute z bounds based on bounding box z_min
    abs_z_min = z_min + (rel_z_min if rel_z_min is not None else 0.0)
    abs_z_max = z_min + rel_z_max if rel_z_max is not None else z_max

    # Clip relative bounds to the absolute bounding box
    abs_z_min = max(abs_z_min, z_min)
    abs_z_max = min(abs_z_max, z_max)

    if abs_z_min > abs_z_max:
        print(
            f"  - Warning: computed relative z-range is empty (abs_z_min={abs_z_min} > abs_z_max={abs_z_max}). Returning empty cloud."
        )
        return np.empty((0, 3)), None if colors is None else np.empty(
            (0, colors.shape[1]), dtype=colors.dtype
        )

    # Build spatial mask using clipped relative z-range (single z-range check)
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= abs_z_min)
        & (points[:, 2] <= abs_z_max)
    )

    # Safety: ensure colors length matches points length
    if colors is not None and len(colors) != len(points):
        print(
            f"  - Warning: color array length ({len(colors)}) does not match points length ({len(points)}). Ignoring colors for this file."
        )
        colors = None

    cropped_points = points[mask]
    cropped_colors = colors[mask] if colors is not None else None
    return cropped_points, cropped_colors


def downsample_point_cloud(points, colors, voxel_size=0.01):
    """Downsamples a point cloud using voxel grid."""
    if len(points) == 0:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_pcd, _ = downsampled_pcd.remove_statistical_outlier(15, std_ratio=2.0)

    down_points = np.asarray(downsampled_pcd.points)
    down_colors = (
        (np.asarray(downsampled_pcd.colors) * 255.0).astype(np.uint8)
        if downsampled_pcd.has_colors()
        else np.full(
            (len(down_points), 3), 128, dtype=np.uint8
        )  # Add gray color if no colors
    )
    return down_points, down_colors


def save_as_ply_open3d(points, colors, output_path):
    """Saves a point cloud as a PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        # Open3D expects colors in float format (0-1), so we normalize from uint8 (0-255).
        # We can do this unconditionally as the input `colors` array from previous
        # steps is consistently in the 0-255 range.
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)


def save_as_las(points, colors, output_path):
    """Saves a point cloud as a LAS file."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.0001, 0.0001, 0.0001])

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if colors is not None:
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    las.write(str(output_path))


def merge_las_files(las_paths):
    """Merges multiple LAS files into a single point cloud."""
    all_points = []
    all_colors = []
    for las_path in las_paths:
        points, colors = load_las_file(str(las_path))
        all_points.append(points)
        all_colors.append(colors)
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors) if all_colors else None
    return merged_points, merged_colors


def process_las_to_ply(src_path, dst_path, scene_name, args):
    """Processes a single scene from LAS to PLY."""
    logger = get_logger()
    logger.info(f"Processing LAS to PLY for: {scene_name}")

    downsample = args.downsample
    voxel_size = args.voxel_size

    json_path = src_path / "Annotations" / "floorplan.json"
    las_dir = (
        src_path / "LAS_Refined"
        if (src_path / "LAS_Refined").exists()
        else src_path / "LAS"
    )

    if not json_path.is_file():
        logger.warning(f"floorplan.json not found at {json_path}. Skipping.")
        return False
    if not las_dir.is_dir():
        logger.warning(f"LAS directory not found at {las_dir}. Skipping.")
        return False

    try:
        las_files = list(las_dir.glob("*.las"))
        if not las_files:
            logger.warning(f"No .las files found in {las_dir}. Skipping.")
            return False
        las_path = las_files[0]
        if len(las_files) > 1:
            if las_dir.joinpath("all.las").exists():
                logger.info(
                    f"Found 'all.las' file in {las_dir}. Using it instead of individual LAS files."
                )
                las_path = las_dir.joinpath("all.las")
            else:
                logger.info(
                    f"Multiple .las files found; merge to one: {[f.name for f in las_files]}"
                )
                merged_points, merged_colors = merge_las_files(las_files)
                # save merged point cloud to las file
                save_as_las(merged_points, merged_colors, las_dir.joinpath("all.las"))
                las_path = las_dir.joinpath("all.las")

        bounding_box = read_floorplan_json(str(json_path))
        logger.info(f"Using LAS file: {las_path.name}")
        logger.debug(f"BoundingBox: {bounding_box}")

        points, colors = load_las_file(str(las_path))

        logger.info(f"Original point count: {len(points)}")

        if args.rel_z_min:
            logger.info(f"Applying relative min height crop: {args.rel_z_min}m")
        if args.rel_z_max:
            logger.info(f"Applying relative max height crop: {args.rel_z_max}m")
        cropped_points, cropped_colors = crop_point_cloud(
            points, colors, bounding_box, args.rel_z_min, args.rel_z_max
        )
        logger.info(f"Cropped point count: {len(cropped_points)}")

        if len(cropped_points) == 0:
            logger.warning("Cropped point cloud is empty. Skipping file save.")
            return False

        if downsample:
            logger.info(f"Downsampling with voxel size: {voxel_size}...")
            cropped_points, cropped_colors = downsample_point_cloud(
                cropped_points, cropped_colors, voxel_size
            )
            logger.info(f"Downsampled point count: {len(cropped_points)}")

        pcd_filename = (
            f"{scene_name}_{voxel_size}.ply" if downsample else f"{scene_name}.ply"
        )
        pcd_output_path = dst_path.parent / "pcd" / pcd_filename
        pcd_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving PLY file to: {pcd_output_path}")
        save_as_ply_open3d(cropped_points, cropped_colors, pcd_output_path)
        logger.info("Successfully saved PLY.")
        return True

    except Exception as e:
        logger.error(f"Error processing {scene_name}: {e}")
        logger.debug(traceback.format_exc())
        return False


def worker_las_to_ply(args_tuple):
    """Wrapper function for multiprocessing pool."""
    return process_las_to_ply(*args_tuple)


# ===================================================================================
# Main Execution Logic
# ===================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="""Unified script for HC3D dataset preprocessing.
        Performs LAS-to-PLY cropping and/or DXF-to-TXT conversion based on a data map.""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # --- General Arguments ---
    parser.add_argument(
        "--data_info",
        type=str,
        required=True,
        help="Path to the data_info.txt CSV file.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Directory to save converted data",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["las2ply", "dxf2txt", "all"],
        help="Task to perform:\n"
        "  las2ply: Crop LAS files to PLY based on bounding box.\n"
        "  dxf2txt: Convert DXF files to SpatialLM text format.\n"
        "  all: Perform both tasks sequentially.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel execution. Defaults to CPU count.",
    )
    # --- Logging Arguments ---
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    log_group.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file. If not specified, logs will only be printed to console.",
    )
    log_group.add_argument(
        "--no_console_log",
        action="store_true",
        help="Disable console logging output.",
    )
    # --- Data Processing Arguments ---
    proc_group = parser.add_argument_group("Data Processing Options")
    proc_group.add_argument(
        "--rel_z_min",
        type=float,
        default=0.5,
        help="Minimum height relative to the bounding box minimum for cropping (applies to both las2ply and dxf2txt).",
    )
    proc_group.add_argument(
        "--rel_z_max",
        type=float,
        default=2.2,
        help="Maximum height relative to the bounding box minimum for cropping (applies to both las2ply and dxf2txt).",
    )

    # --- LAS to PLY Arguments ---
    las_group = parser.add_argument_group("LAS to PLY Options")
    las_group.add_argument(
        "--downsample",
        action="store_true",
        help="Enable voxel downsampling for the point cloud.",
    )
    las_group.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling (default: 0.01).",
    )

    # --- DXF to TXT Arguments ---
    dxf_group = parser.add_argument_group("DXF to TXT Options")
    dxf_group.add_argument(
        "--dxf_filename",
        type=str,
        default="3D.dxf",
        help="Name of the DXF file to look for (default: 3D.dxf).",
    )
    dxf_group.add_argument(
        "--extrude_from_floorplan",
        action="store_true",
        help="Enable 2D floorplan extrusion mode. This will look for 'floorplan.dxf' and requires --max_height.",
    )
    dxf_group.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Geometric tolerance for matching walls to doors/windows (default: 1e-3).",
    )
    dxf_group.add_argument(
        "--min-wall-width",
        type=float,
        default=0.02,
        help="Minimum wall width to consider (default: 0.02m).",
    )
    dxf_group.add_argument(
        "--point-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for removing redundant points in DXF polylines (default: 1e-6).",
    )
    dxf_group.add_argument(
        "--no-point-cleanup",
        action="store_true",
        help="Disable redundant point removal in DXF processing.",
    )

    args = parser.parse_args()

    # --- Initialize Logging System ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = args.log_file
    if log_file is None and args.dst_dir:
        # 如果没有指定日志文件，则在输出目录中创建默认日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(args.dst_dir) / f"preprocess_hc3d_{timestamp}.log"

    setup_logger(
        name="preprocess_hc3d",
        log_level=log_level,
        log_file=log_file,
        console=not args.no_console_log,
    )
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("HC3D Dataset Preprocessing Script Started")
    logger.info("=" * 70)
    logger.info(f"Log level: {args.log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    logger.info(f"Task(s) to execute: {args.task}")
    logger.info(f"Data info file: {args.data_info}")
    logger.info(f"Output directory: {args.dst_dir}")

    # --- Data Info Loading ---
    data_info_path = Path(args.data_info)
    if not data_info_path.is_file():
        parser.error(f"Data map file not found: {data_info_path}")

    logger.info(f"Loading data info from: {data_info_path}")

    try:
        df = parse_csv(args.data_info)
        if "path" not in df.columns or "scene_id" not in df.columns:
            parser.error("Data map CSV must contain 'path' and 'scene_id' columns.")
        logger.info(f"Successfully loaded {len(df)} items from data info file")
    except Exception as e:
        parser.error(f"Failed to read or parse data map file: {e}")

    # --- Task Execution ---
    num_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    logger.info(
        f"Starting task(s): '{args.task.upper()}' on {len(df)} items from '{data_info_path.name}'"
    )
    logger.info(f"Using {num_workers} worker processes for parallelizable tasks.")

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {dst_dir}")

    if args.task in ["las2ply", "all"]:
        logger.info("=" * 70)
        logger.info("Executing task: LAS2PLY")
        logger.info("=" * 70)

        tasks = [
            (
                Path(row["path"]),
                dst_dir.joinpath(f"scene_{row['scene_id']:05d}"),
                f"scene_{row['scene_id']:05d}",
                args,
            )
            for _, row in df.iterrows()
        ]
        logger.info(f"Created {len(tasks)} LAS2PLY processing tasks")

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_las_to_ply, tasks),
                    total=len(tasks),
                    desc="Processing LAS to PLY",
                )
            )

        success_count = sum(1 for r in results if r)
        logger.info("-" * 70)
        logger.info(
            f"Task 'LAS2PLY' complete. Successfully processed {success_count}/{len(df)} items."
        )

    if args.task in ["dxf2txt", "all"]:
        logger.info("=" * 70)
        logger.info("Executing task: DXF2TXT")
        logger.info("=" * 70)

        success_count = 0
        # For tqdm, we convert iterrows() to a list to get the total count
        for index, row in tqdm(list(df.iterrows()), desc="Processing DXF to TXT"):
            src_path = Path(row["path"])
            scene_name = f"scene_{row['scene_id']:05d}"
            dst_path = dst_dir.joinpath(scene_name)
            scene_name = dst_path.name

            logger.debug(f"Processing item {index + 1}/{len(df)}: {scene_name}")
            if args.rel_z_min:
                logger.debug(f"Applying relative min height crop: {args.rel_z_min}m")
            if args.rel_z_max:
                logger.debug(f"Applying relative max height crop: {args.rel_z_max}m")

            if process_dxf_to_txt(src_path, dst_path, scene_name, args):
                success_count += 1

        logger.info("-" * 70)
        logger.info(
            f"Task 'DXF2TXT' complete. Successfully processed {success_count}/{len(df)} items."
        )

    logger.info("=" * 70)
    logger.info("All tasks finished.")


if __name__ == "__main__":
    main()
