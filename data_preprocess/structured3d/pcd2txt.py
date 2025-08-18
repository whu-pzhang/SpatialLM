# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

MM2M = 1.0 / 1000.0
TOL = 1e-3  # meters


@dataclass
class Wall:
    """墙体数据结构"""

    ax: float  # 底边起点x
    ay: float  # 底边起点y
    az: float  # 底边起点z
    bx: float  # 底边终点x
    by: float  # 底边终点y
    bz: float  # 底边终点z
    height: float  # 墙体高度
    thickness: float  # 墙厚
    id: Optional[int] = None  # 墙体ID
    plane_id: Optional[int] = None  # 平面ID
    room_type: Optional[str] = None  # 所属房间类型

    def __str__(self):
        # 格式化输出，限制小数位数
        return f"Wall({self.ax:.9g},{self.ay:.9g},{self.az:.9g},{self.bx:.9g},{self.by:.9g},{self.bz:.9g},{self.height:.9g},{int(self.thickness)})"


@dataclass
class Door:
    """门数据结构"""

    wall_id: int  # 挂靠的墙体ID
    cx: float  # 中心x
    cy: float  # 中心y
    cz: float  # 中心z
    width: float  # 宽度
    height: float  # 高度

    def __str__(self):
        # 格式化输出，限制小数位数
        return f"Door(wall_{self.wall_id},{self.cx:.9g},{self.cy:.9g},{self.cz:.9g},{self.width:.9g},{self.height:.9g})"


@dataclass
class Window:
    """窗数据结构"""

    wall_id: int  # 挂靠的墙体ID
    cx: float  # 中心x
    cy: float  # 中心y
    cz: float  # 中心z
    width: float  # 宽度
    height: float  # 高度

    def __str__(self):
        # 格式化输出，限制小数位数
        return f"Window(wall_{self.wall_id},{self.cx:.9g},{self.cy:.9g},{self.cz:.9g},{self.width:.9g},{self.height:.9g})"


@dataclass
class Opening:
    """门窗的临时几何数据结构"""

    cx: float
    cy: float
    cz: float
    width: float
    height: float
    normal: np.ndarray


def load_json(path="annotation_d.json"):
    """加载JSON文件"""
    with open(path, "r") as f:
        return json.load(f)


def extract_rooms(annos: Dict) -> Dict:
    """提取所有房间及其组成平面"""
    rooms = {}

    for semantic in annos["semantics"]:
        sem_type = semantic["type"]

        # 跳过门窗和外墙
        if sem_type in ["door", "window", "outwall"]:
            continue

        if sem_type not in rooms:
            rooms[sem_type] = []

        room_data = {
            "semantic_id": semantic.get("ID", -1),
            "type": sem_type,
            "plane_ids": semantic["planeID"],
            "floor_planes": [],
            "ceiling_planes": [],
            "wall_planes": [],
        }

        # 分类平面
        for pid in semantic["planeID"]:
            plane = annos["planes"][pid]
            plane_type = plane["type"]

            if plane_type == "floor":
                room_data["floor_planes"].append(pid)
            elif plane_type == "ceiling":
                room_data["ceiling_planes"].append(pid)
            elif plane_type == "wall":
                room_data["wall_planes"].append(pid)

        rooms[sem_type].append(room_data)

    return rooms


def get_plane_z_coordinate(plane_id: int, annos: Dict) -> float:
    """获取平面的Z坐标（用于地板和天花板）"""
    plane = annos["planes"][plane_id]

    # 获取平面的所有线
    PLM = np.array(annos["planeLineMatrix"], dtype=int)
    line_ids = np.where(PLM[plane_id])[0]

    # 获取所有端点的Z坐标
    z_coords = []
    LJM = np.array(annos["lineJunctionMatrix"], dtype=int)

    for line_id in line_ids:
        junction_ids = np.where(LJM[line_id])[0]
        for jid in junction_ids:
            junction = annos["junctions"][jid]
            z_coords.append(junction["coordinate"][2] * MM2M)

    # 返回平均Z坐标
    return np.mean(z_coords) if z_coords else 0.0


def calculate_room_height(room: Dict, annos: Dict) -> Tuple[float, float]:
    """计算房间的地板高度和房间高度"""
    z_floor = float("inf")
    z_ceiling = float("-inf")

    # 获取地板高度
    for pid in room["floor_planes"]:
        z = get_plane_z_coordinate(pid, annos)
        z_floor = min(z_floor, z)

    # 获取天花板高度
    for pid in room["ceiling_planes"]:
        z = get_plane_z_coordinate(pid, annos)
        z_ceiling = max(z_ceiling, z)

    # 如果没有找到地板或天花板，使用默认值
    if z_floor == float("inf"):
        z_floor = 0.0
    if z_ceiling == float("-inf"):
        z_ceiling = 2.8  # 默认高度

    height = z_ceiling - z_floor

    return z_floor, height


def get_wall_endpoints(plane_id: int, annos: Dict) -> List[Tuple[float, float, float]]:
    """获取墙体平面的所有端点"""
    PLM = np.array(annos["planeLineMatrix"], dtype=int)
    LJM = np.array(annos["lineJunctionMatrix"], dtype=int)

    # 获取平面的所有线
    line_ids = np.where(PLM[plane_id])[0]

    # 收集所有端点
    points = []
    for line_id in line_ids:
        junction_ids = np.where(LJM[line_id])[0]
        for jid in junction_ids:
            junction = annos["junctions"][jid]
            coord = junction["coordinate"]
            points.append((coord[0] * MM2M, coord[1] * MM2M, coord[2] * MM2M))

    return points


def extract_wall_geometry(
    plane_id: int, annos: Dict, z_floor: float, height: float, room_type: str
) -> Wall:
    """从平面ID提取墙体的几何信息"""
    # 获取墙体的所有端点
    points = get_wall_endpoints(plane_id, annos)

    if len(points) < 2:
        return None

    # 将点投影到XY平面并去重
    xy_points = [(p[0], p[1]) for p in points]
    unique_points = list(set(xy_points))

    # 如果墙体是垂直的，找到两个最远的点作为底边端点
    if len(unique_points) >= 2:
        # 计算所有点对之间的距离，找到最远的两个点
        max_dist = 0
        best_pair = (unique_points[0], unique_points[1])

        for i in range(len(unique_points)):
            for j in range(i + 1, len(unique_points)):
                dist = np.sqrt(
                    (unique_points[i][0] - unique_points[j][0]) ** 2
                    + (unique_points[i][1] - unique_points[j][1]) ** 2
                )
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (unique_points[i], unique_points[j])

        # 创建墙体对象
        wall = Wall(
            ax=best_pair[0][0],
            ay=best_pair[0][1],
            az=z_floor,
            bx=best_pair[1][0],
            by=best_pair[1][1],
            bz=z_floor,
            height=height,
            thickness=0.0,  # 默认厚度
            plane_id=plane_id,
            room_type=room_type,
        )

        return wall

    return None


def extract_opening_geometry(semantic: Dict, annos: Dict) -> Optional[Opening]:
    """
    提取门窗的几何信息, 包括精确宽度和法向量
    """
    plane_ids = semantic["planeID"]
    planes = [annos["planes"][pid] for pid in plane_ids]

    # 1. 区分"上下"平面和"左右"平面
    up_vector = np.array([0, 0, 1])
    horizontal_planes = []  # 上下
    vertical_planes = []  # 左右

    for plane in planes:
        normal = np.array(plane["normal"][:3])
        # 如果法向量与UP向量的点积绝对值接近1, 则为水平面
        if abs(np.dot(normal, up_vector)) > 0.95:
            horizontal_planes.append(plane)
        else:
            vertical_planes.append(plane)

    # 如果没有找到两对平行的平面(例如数据不规范), 则回退
    if len(vertical_planes) < 2 or len(horizontal_planes) < 2:
        return _extract_opening_geometry_fallback(semantic, annos)

    side_planes = vertical_planes

    # 2. 精确计算宽度
    # 宽度是两个"左右"(vertical)平面的offset之和
    width = abs(side_planes[0]["offset"] + side_planes[1]["offset"]) * MM2M

    # 3. 计算高度和中心点 (维持旧逻辑)
    all_points = []
    for pid in plane_ids:
        points = get_wall_endpoints(pid, annos)
        all_points.extend(points)

    if not all_points:
        return None

    z_coords = [p[2] for p in all_points]
    min_z, max_z = min(z_coords), max(z_coords)
    height = max_z - min_z

    # 使用所有角点的包围盒中心来计算
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2

    # 4. 计算法向量
    # jamb 平面的法向量是互相平行的, 且垂直于开口的主方向
    # 通过两个jamb法向量的叉乘,可以得到门窗的up向量,再通过jamb法向量得到门窗的法向量
    n1 = np.array(side_planes[0]["normal"][:3])

    # 开口的主法向量应该垂直于 侧面 的法向量, 同时垂直于 "UP" 向量
    up_vector = np.array([0, 0, 1])
    opening_normal = np.cross(n1, up_vector)

    # 防止法向量为0(例如,侧面是水平的)
    if np.linalg.norm(opening_normal) < TOL:
        # 如果侧面是水平的, 那么开口法向量就是UP向量
        opening_normal = up_vector
    else:
        opening_normal = opening_normal / np.linalg.norm(opening_normal)

    return Opening(cx, cy, cz, width, height, opening_normal)


def _extract_opening_geometry_fallback(
    semantic: Dict, annos: Dict
) -> Optional[Opening]:
    """对于无jamb平面的开口, 使用包围盒方法作为后备"""
    plane_ids = semantic["planeID"]
    all_points = []
    for pid in plane_ids:
        points = get_wall_endpoints(pid, annos)
        all_points.extend(points)
    if not all_points:
        return None

    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    z_coords = [p[2] for p in all_points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2

    # 在不知道jamb的情况下, 宽度只能估算
    width = max(max_x - min_x, max_y - min_y)
    height = max_z - min_z

    # 法向量可以通过查找虚拟平面获得
    opening_normal = np.array([1, 0, 0])  # 默认X朝向
    for pid in plane_ids:
        # 有些door/window包含一个代表开口本身的平面
        plane = annos["planes"][pid]
        if plane["type"] in ["door", "window"]:
            opening_normal = np.array(plane["normal"][:3])
            break

    return Opening(cx, cy, cz, width, height, opening_normal)


def find_adjacent_walls(
    opening_semantic: Dict, walls: List[Wall], annos: Dict
) -> List[Wall]:
    """通过共享线来查找与门窗邻接的墙体"""
    opening_plane_ids = set(opening_semantic["planeID"])
    adjacent_walls = []

    # 预计算墙体plane_id集合, 提高查询效率
    wall_plane_ids = {wall.plane_id: wall for wall in walls}

    # 遍历门窗的所有平面
    for pid in opening_plane_ids:
        # 获取当前平面的所有线
        PLM = np.array(annos["planeLineMatrix"], dtype=int)
        line_ids = np.where(PLM[pid])[0]

        # 遍历每条线, 查找共享这条线的其他平面
        for line_id in line_ids:
            shared_plane_ids = np.where(PLM[:, line_id])[0]

            for shared_pid in shared_plane_ids:
                if shared_pid not in opening_plane_ids and shared_pid in wall_plane_ids:
                    # 这个平面是墙体, 且不属于这个门窗自身
                    wall = wall_plane_ids[shared_pid]
                    if wall not in adjacent_walls:
                        adjacent_walls.append(wall)

    return adjacent_walls


def align_opening_to_wall(opening: Opening, wall: Wall) -> Tuple[float, float, float]:
    """将门窗中心投影到墙体平面上, 确保共面性"""
    # 1. 定义墙体平面
    p1 = np.array([wall.ax, wall.ay, wall.az])
    p2 = np.array([wall.bx, wall.by, wall.bz])
    p3 = np.array([wall.ax, wall.ay, wall.az + wall.height])

    # Check for degenerate cases (zero-length wall segment)
    if np.linalg.norm(p2 - p1) < TOL or np.linalg.norm(p3 - p1) < TOL:
        return opening.cx, opening.cy, opening.cz

    wall_normal = np.cross(p2 - p1, p3 - p1)

    # Check if normal is valid
    if np.linalg.norm(wall_normal) < TOL:
        return opening.cx, opening.cy, opening.cz

    wall_normal = wall_normal / np.linalg.norm(wall_normal)

    # D in Ax+By+Cz+D=0
    d = -np.dot(wall_normal, p1)

    # 2. 校验法向量
    # 如果门窗法向量和墙体法向量点积非~1或~-1, 说明二者不平行, 挂靠可能错误
    if abs(abs(np.dot(opening.normal, wall_normal)) - 1.0) > 0.1:
        # print(f"Warning: Opening normal {opening.normal} is not parallel to wall normal {wall_normal}. Skipping projection.")
        return opening.cx, opening.cy, opening.cz

    # 3. 计算投影
    x0, y0, z0 = opening.cx, opening.cy, opening.cz
    t = -(wall_normal[0] * x0 + wall_normal[1] * y0 + wall_normal[2] * z0 + d)

    proj_x = x0 + t * wall_normal[0]
    proj_y = y0 + t * wall_normal[1]
    proj_z = z0 + t * wall_normal[2]

    return proj_x, proj_y, proj_z


def point_to_line_segment_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """计算点到线段的距离"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 线段的向量
    dx = x2 - x1
    dy = y2 - y1

    # 线段长度的平方
    line_length_sq = dx * dx + dy * dy

    if line_length_sq == 0:
        # 线段退化为点
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # 计算投影参数
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_length_sq))

    # 投影点
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    # 返回距离
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def extract_layout(annos: Dict) -> Tuple[List[Wall], List[Door], List[Window]]:
    """主函数：提取布局信息"""
    walls = []
    doors = []
    windows = []

    # 1. 提取房间信息
    rooms = extract_rooms(annos)
    # print(f"找到 {len(rooms)} 种房间类型")

    # 2. 处理每个房间
    for room_type, room_list in rooms.items():
        for room in room_list:
            # print(f"处理房间: {room_type}")

            # 3. 计算房间高度
            z_floor, height = calculate_room_height(room, annos)
            # print(f"  地板高度: {z_floor:.3f}m, 房间高度: {height:.3f}m")

            # 4. 提取墙体
            for wall_plane_id in room["wall_planes"]:
                wall = extract_wall_geometry(
                    wall_plane_id, annos, z_floor, height, room_type
                )
                if wall:
                    wall.id = len(walls)
                    walls.append(wall)

            # print(f"  提取了 {len(room['wall_planes'])} 面墙")

    # 5. 提取门窗
    for semantic in annos["semantics"]:
        sem_type = semantic["type"]

        if sem_type == "door":
            opening_geom = extract_opening_geometry(semantic, annos)
            if opening_geom:
                # 使用新的邻接逻辑查找挂靠的墙体
                attached_walls = find_adjacent_walls(semantic, walls, annos)
                for wall in attached_walls:
                    # 坐标对齐
                    cx, cy, cz = align_opening_to_wall(opening_geom, wall)
                    door = Door(
                        wall_id=wall.id,
                        cx=cx,
                        cy=cy,
                        cz=cz,
                        width=opening_geom.width,
                        height=opening_geom.height,
                    )
                    doors.append(door)

        elif sem_type == "window":
            opening_geom = extract_opening_geometry(semantic, annos)
            if opening_geom:
                # 使用新的邻接逻辑查找挂靠的墙体
                attached_walls = find_adjacent_walls(semantic, walls, annos)
                for wall in attached_walls:
                    # 坐标对齐
                    cx, cy, cz = align_opening_to_wall(opening_geom, wall)
                    window = Window(
                        wall_id=wall.id,
                        cx=cx,
                        cy=cy,
                        cz=cz,
                        width=opening_geom.width,
                        height=opening_geom.height,
                    )
                    windows.append(window)

    # print(f"\n总计: {len(walls)} 面墙, {len(doors)} 扇门, {len(windows)} 扇窗")

    return walls, doors, windows


def format_output(
    walls: List[Wall],
    doors: List[Door],
    windows: List[Window],
    include_room_labels: bool = False,
) -> str:
    """格式化输出结果

    Args:
        walls: 墙体列表
        doors: 门列表
        windows: 窗列表
        include_room_labels: 是否包含房间标注
    """
    lines = []

    # 输出墙体
    for i, wall in enumerate(walls):
        line = f"wall_{i}={wall}"
        # 可选：添加房间标注
        if include_room_labels and wall.room_type:
            line += f"  # {wall.room_type}"
        lines.append(line)

    # 输出门
    for i, door in enumerate(doors):
        lines.append(f"door_{i}={door}")

    # 输出窗
    for i, window in enumerate(windows):
        lines.append(f"window_{i}={window}")

    return "\n".join(lines)


def save_output(content: str, filename: str = "output.txt"):
    """保存输出到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from tqdm import tqdm

    # --- Main execution block ---
    parser = argparse.ArgumentParser(
        description="从Structured3D格式的JSON文件中提取墙体、门窗布局, 支持批量处理."
    )
    parser.add_argument(
        "--input-dir",
        default="data/Structured3D/Panorama",
        required=True,
        help="path to raw Structured3D_panorama directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        default="data/structured3d-spatiallm/layout_custom",
        help="output directory.",
    )
    parser.add_argument(
        "--room-labels", action="store_true", help="在输出中包含房间类型标注."
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细的运行时信息.")

    args = parser.parse_args()

    # 查找所有目标json文件
    # search_pattern = os.path.join(args.input_dir, "**", "annotation_3d.json")
    # json_files = glob.glob(search_pattern, recursive=True)
    data_root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            scenes = [
                f
                for f in item.joinpath("Structured3D").iterdir()
                if f.is_dir() and "scene" in f.name
            ]
            scene_dirs.extend(scenes)
    print(f"找到了 {len(scene_dirs)} 个场景文件. 开始处理...")

    total_walls, total_doors, total_windows = 0, 0, 0

    for scene_dir in tqdm(scene_dirs):
        output_filename = scene_dir.name + ".txt"
        output_path = output_dir.joinpath(output_filename)
        json_path = scene_dir.joinpath("annotation_3d.json")

        try:
            # 1. 加载数据
            annos = load_json(json_path)

            # 2. 提取布局
            if args.verbose:
                print("  提取布局信息...")
            walls, doors, windows = extract_layout(annos)

            total_walls += len(walls)
            total_doors += len(doors)
            total_windows += len(windows)

            # 3. 格式化输出
            output_content = format_output(
                walls, doors, windows, include_room_labels=args.room_labels
            )

            # 4. 保存到文件
            save_output(output_content, output_path)

            if args.verbose:
                print("  统计信息:")
                print(f"    墙体: {len(walls)}, 门: {len(doors)}, 窗: {len(windows)}")

        except Exception as e:
            print(f"  *** 处理文件时发生严重错误: {e} ***")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有场景处理完毕.")
    print(f"总计: {total_walls} 面墙, {total_doors} 扇门, {total_windows} 扇窗.")
