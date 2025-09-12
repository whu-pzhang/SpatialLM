"""
在 DXF 文件中，每个房间、门和窗的实例分别存放在不同的图层中。
包含有 room, door, window, bay_window 和 ordinary_window 的图层，
不同图层命名采用 room_id, door_id 和 window_id 进行区分。
每个三维实体的几何形状由多个 LWPOLYLINE 和 POLYLINE 共同构成。

先需要将这些三维实体转换为 SpatialLM 所需的文本格式，只包含 wall, door 和 window。要求如下：
1. wall: 由两个三维点 (ax, ay, az), (bx, by, bz) 定义墙体的底边，height 定义墙体高度，thickness 定义墙厚, 厚度目前统一取零
2. door: 由墙体 wall_id 挂靠，中心点 (cx, cy, cz)，width 定义宽度，height 定义高度，与挂靠墙体共面
3. window: 由墙体 wall_id 挂靠，中心点 (cx, cy, cz)，width 定义宽度，height 定义高度，与挂靠墙体共面

注意：不需要表示地面和天花板。
简化版本：按用户要求的顺序处理点清理。

Wall, Door, Window 数据结构已定义好。
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import ezdxf
    import numpy as np
    from ezdxf.math import OCS, Vec3

    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    print("Warning: ezdxf and numpy are not available.")


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


@dataclass
class Entity:
    type: str
    id: str
    points: List[tuple] = field(default_factory=list)
    layer: Optional[str] = None


def get_points_from_lwpolyline(e):
    elev = float(getattr(e.dxf, "elevation", 0.0))
    extrusion = getattr(e.dxf, "extrusion", Vec3(0, 0, 1))
    ocs = OCS(extrusion)
    pts = [ocs.to_wcs((float(raw[0]), float(raw[1]), elev)) for raw in e.get_points()]
    return [(p.x, p.y, p.z) for p in pts]


def remove_redundant_points(points, tolerance=1e-1):
    """
    按用户要求的顺序移除多余的点：
    1. 先移除重复点
    2. 移除共线点中的冗余

    Args:
        points: 点列表 [(x, y, z), ...]
        tolerance: 距离容差，默认1微米

    Returns:
        cleaned_points: 清理后的点列表
    """
    if len(points) <= 2:
        return points

    points = np.array(points)

    # 第一步：移除重复点
    points = remove_duplicate_points(points, tolerance)

    if len(points) <= 2:
        return [tuple(p) for p in points]

    # 第二步：移除共线点中的冗余
    points = remove_collinear_points(points, tolerance)

    return [tuple(p) for p in points]


def remove_duplicate_points(points, tolerance=1e-6):
    """
    移除重复点（包括首尾相同的点和中间的重复点）

    Args:
        points: 点列表
        tolerance: 距离容差

    Returns:
        cleaned_points: 移除重复点后的点列表
    """
    if len(points) <= 1:
        return points

    points = np.array(points)

    # 先移除首尾相同的点（闭合多边形处理）
    while len(points) > 2:
        first_point = points[0]
        last_point = points[-1]
        if np.linalg.norm(last_point - first_point) < tolerance:
            points = points[:-1]  # 移除最后一个点
        else:
            break

    # 再移除所有重复点和过近点
    unique_points = []
    for point in points:
        is_duplicate = False
        for existing_point in unique_points:
            if np.linalg.norm(point - existing_point) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)

    return np.array(unique_points)


def remove_collinear_points(points, tolerance=1e-6):
    """
    移除共线点中的冗余

    Args:
        points: 点列表
        tolerance: 共线判断容差

    Returns:
        simplified_points: 移除共线点后的点列表
    """
    if len(points) <= 2:
        return points

    points = np.array(points)

    # 反复迭代移除共线点，直到无法再简化
    changed = True
    while changed and len(points) > 2:
        changed = False
        new_points = []

        for i in range(len(points)):
            # 获取前一个点和后一个点的索引
            prev_i = (i - 1) % len(points)
            next_i = (i + 1) % len(points)

            # 检查是否为共线点
            prev_point = points[prev_i]
            curr_point = points[i]
            next_point = points[next_i]

            # 计算当前点到前后两点连线的距离
            dist_to_line = point_to_line_distance(curr_point, prev_point, next_point)

            if dist_to_line > tolerance:
                # 不是共线点，保留
                new_points.append(curr_point)
            else:
                # 是共线点，标记为已改变
                changed = True

        if changed and len(new_points) > 2:
            points = np.array(new_points)
        elif not changed:
            break

    return points


def point_to_line_distance(point, line_start, line_end):
    """
    计算点到线段的距离

    Args:
        point: 目标点
        line_start: 线段起点
        line_end: 线段终点

    Returns:
        distance: 点到线段的距离
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length < 1e-12:
        # 线段长度为0，返回点到起点的距离
        return np.linalg.norm(point - line_start)

    # 计算点在线段上的投影
    point_vec = point - line_start
    projection_length = np.dot(point_vec, line_vec) / line_length
    projection_ratio = projection_length / line_length

    if projection_ratio < 0:
        # 投影在线段起点之前
        return np.linalg.norm(point - line_start)
    elif projection_ratio > 1:
        # 投影在线段终点之后
        return np.linalg.norm(point - line_end)
    else:
        # 投影在线段上
        projection_point = line_start + projection_ratio * line_vec
        return np.linalg.norm(point - projection_point)


def parse_layer_name(layer_name):
    """
    解析图层名称，正确区分不同类型的窗户
    """
    name_lower = layer_name.lower()

    if name_lower.startswith("room"):
        entity_type = "room"
        entity_id = name_lower
    elif name_lower.startswith("door"):
        entity_type = "door"
        entity_id = name_lower
    elif "window" in name_lower:
        entity_type = "window"
        entity_id = name_lower
    else:
        return None, None

    return entity_type, entity_id


def is_coplanar_and_center_in_wall(face, wall, tolerance=0.1):
    """
    检查门窗面是否与墙体匹配（简化版本）
    只需要满足两个条件：
    1. 共面（法向量平行且距离很近）
    2. 门窗中心点在墙体面中
    """
    # 1. 检查是否共面
    normal_dot = abs(np.dot(wall.normal, face["normal"]))
    if abs(normal_dot - 1.0) > tolerance:
        return False, float("inf")

    # 2. 检查门窗中心点是否在墙体面中
    wall_start = np.array([wall.ax, wall.ay])
    wall_end = np.array([wall.bx, wall.by])
    wall_direction = wall_end - wall_start
    wall_length = np.linalg.norm(wall_direction)

    if wall_length < tolerance:
        return False, float("inf")

    wall_direction_normalized = wall_direction / wall_length

    # 计算门窗中心点在墙体方向上的投影
    face_center_2d = np.array([face["center"][0], face["center"][1]])
    face_to_wall_start = face_center_2d - wall_start

    # 门窗中心在墙体方向上的投影位置
    center_projection = np.dot(face_to_wall_start, wall_direction_normalized)

    # 检查中心点是否在墙体范围内
    if not (0 <= center_projection <= wall_length):
        return False, float("inf")

    # 3. 检查距离
    wall_normal_2d = np.array(
        [-wall_direction_normalized[1], wall_direction_normalized[0]]
    )
    distance_to_wall = abs(np.dot(face_to_wall_start, wall_normal_2d))

    if distance_to_wall > tolerance:
        return False, distance_to_wall

    return True, distance_to_wall


def infer_vertical_faces(entity_list, min_wall_width=0.01, min_wall_height=0.08):
    """从DXF实体列表中推断出所有竖直面"""
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
    bottom_z, top_z = sorted_zs[0], sorted_zs[-1]
    height = abs(top_z - bottom_z)

    # 降低最小墙体高度要求，避免遗漏真实但较矮的墙体
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

            # 对墙体宽度采用更宽松的限制，避免遗漏真实但较窄的墙体
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


def find_matching_wall(face, walls, tolerance=1e-1):
    """为门窗面找到匹配的墙体"""
    best_wall = None
    best_distance = float("inf")

    for wall in walls:
        is_matching, distance = is_coplanar_and_center_in_wall(face, wall, tolerance)

        if is_matching and distance < best_distance:
            best_wall = wall
            best_distance = distance

    return best_wall, best_distance


def process_dxf_file(input_file, output_file, args):
    """Processes a single DXF file."""
    if not EZDXF_AVAILABLE:
        return

    try:
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()
    except IOError:
        print(f"Error reading file: {input_file}")
        return
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupted DXF file: {input_file}")
        return

    entities = defaultdict(list)
    for e in msp.query("LWPOLYLINE POLYLINE"):
        entity_type, entity_id = parse_layer_name(e.dxf.layer)
        if not entity_type:
            continue
        points = (
            get_points_from_lwpolyline(e)
            if e.dxftype() == "LWPOLYLINE"
            else [v.dxf.location for v in e.vertices]
        )

        if not args.no_point_cleanup:
            points = remove_redundant_points(points, args.point_tolerance)

        entities[(entity_type, entity_id)].append(
            Entity(type=entity_type, id=entity_id, points=points, layer=e.dxf.layer)
        )

    walls, doors, windows = [], [], []
    wall_id_counter = 0

    # 1. Extract walls
    for (etype, eid), entity_list in entities.items():
        if etype != "room":
            continue
        faces = infer_vertical_faces(
            entity_list, args.min_wall_width, 0.01
        )  # 设置最小墙体高度为1cm
        for face in faces:
            p1, p2 = face["bottom_p1"], face["bottom_p2"]
            walls.append(
                Wall(
                    ax=p1[0],
                    ay=p1[1],
                    az=p1[2],
                    bx=p2[0],
                    by=p2[1],
                    bz=p2[2],
                    height=face["height"],
                    thickness=0.0,
                    id=wall_id_counter,
                    room_type=f"room_{eid}",
                    normal=face["normal"],
                )
            )
            wall_id_counter += 1

    # 2. Extract and associate doors and windows
    for (etype, eid), entity_list in entities.items():
        if etype not in ["door", "window"]:
            continue
        min_width_for_doors_windows = max(
            args.min_wall_width * 0.2, 0.01
        )  # 门窗宽度最小为墙体宽度的20%或1cm
        faces = infer_vertical_faces(entity_list, min_width_for_doors_windows, 0.01)

        for face in faces:
            best_wall, _ = find_matching_wall(face, walls, args.tolerance)
            if best_wall is not None:
                target_list = doors if etype == "door" else windows
                target_list.append(
                    globals()[etype.capitalize()](
                        wall_id=best_wall.id,
                        cx=face["center"][0],
                        cy=face["center"][1],
                        cz=face["center"][2],
                        width=face["width"],
                        height=face["height"],
                    )
                )

    # 3. Write output
    with open(output_file, "w", encoding="utf-8") as f:
        walls.sort(key=lambda w: w.id)
        for w in walls:
            f.write(w.to_language_string() + "\n")

        door_id_counter = 0
        for d in doors:
            f.write(d.to_language_string(door_id_counter) + "\n")
            door_id_counter += 1

        window_id_counter = 0
        for w in windows:
            f.write(w.to_language_string(window_id_counter) + "\n")
            window_id_counter += 1

    print(f"Processed '{input_file}' -> '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DXF 3D models to SpatialLM text format (Simple Version)."
    )
    parser.add_argument("-i", "--input", type=str, help="Input file DXF file.")
    parser.add_argument(
        "--input_dir", type=str, help="Input file directory with DXF files."
    )
    parser.add_argument("-o", "--output", type=str, help="Output file or directory.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Geometric tolerance for matching.",
    )
    parser.add_argument(
        "--min-wall-width",
        type=float,
        default=0.02,
        help="Minimum wall width in meters.",
    )
    parser.add_argument(
        "--point-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for removing redundant points.",
    )
    parser.add_argument(
        "--no-point-cleanup",
        action="store_true",
        help="Disable redundant point removal.",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required.")

    if args.input and args.input_dir:
        parser.error("Provide either --input or --input-dir, not both.")

    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory not found at '{args.input_dir}'")
            return

        output_dir = args.output or os.path.join(
            os.path.dirname(args.input_dir), "output_simple"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(".dxf"):
                input_file = os.path.join(args.input_dir, filename)
                output_file = os.path.join(
                    output_dir, os.path.splitext(filename)[0] + ".txt"
                )
                process_dxf_file(input_file, output_file, args)
    else:
        output_file = args.output or "output_simple.txt"
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, Path(args.input).stem + ".txt")
        else:
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        process_dxf_file(args.input, output_file, args)


if __name__ == "__main__":
    main()
