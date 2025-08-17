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


def load_json(path="annotation_3d.json"):
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


def extract_opening_geometry(
    semantic: Dict, annos: Dict
) -> Tuple[float, float, float, float, float]:
    """提取门窗的几何信息"""
    plane_ids = semantic["planeID"]

    # 收集所有角点
    all_points = []
    for pid in plane_ids:
        points = get_wall_endpoints(pid, annos)
        all_points.extend(points)

    if not all_points:
        return None

    # 提取所有坐标
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    z_coords = [p[2] for p in all_points]

    # 计算边界框
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    # 计算中心和尺寸
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2

    # 计算宽度（取X和Y方向的最大值作为宽度）
    width = max(max_x - min_x, max_y - min_y)
    height = max_z - min_z

    return cx, cy, cz, width, height


def find_attached_walls(
    opening_semantic: Dict, walls: List[Wall], annos: Dict
) -> List[int]:
    """找到门窗挂靠的所有墙体（门窗通常连接两个房间，因此会挂靠到两面墙）"""
    opening_plane_ids = opening_semantic["planeID"]

    # 获取门窗的中心位置和尺寸
    opening_data = extract_opening_geometry(opening_semantic, annos)
    if not opening_data:
        return []

    cx, cy, cz, width, height = opening_data

    # 找到所有距离很近的墙体
    attached_walls = []
    threshold = 0.15  # 阈值，用于判断门窗是否挂靠在墙上

    for idx, wall in enumerate(walls):
        # 计算门窗中心到墙体的距离
        dist = point_to_line_segment_distance(
            (cx, cy), (wall.ax, wall.ay), (wall.bx, wall.by)
        )

        # 如果距离小于阈值，认为门窗挂靠在这面墙上
        if dist < threshold:
            attached_walls.append((idx, dist))

    # 按距离排序
    attached_walls.sort(key=lambda x: x[1])

    # 通常门窗会挂靠在2面墙上（连接两个房间）
    # 返回最近的2面墙（如果有的话）
    result = []
    for wall_idx, dist in attached_walls[:2]:  # 最多取2面墙
        result.append(wall_idx)

    return result


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
    print(f"找到 {len(rooms)} 种房间类型")

    # 2. 处理每个房间
    for room_type, room_list in rooms.items():
        for room in room_list:
            print(f"处理房间: {room_type}")

            # 3. 计算房间高度
            z_floor, height = calculate_room_height(room, annos)
            print(f"  地板高度: {z_floor:.3f}m, 房间高度: {height:.3f}m")

            # 4. 提取墙体
            for wall_plane_id in room["wall_planes"]:
                wall = extract_wall_geometry(
                    wall_plane_id, annos, z_floor, height, room_type
                )
                if wall:
                    wall.id = len(walls)
                    walls.append(wall)

            print(f"  提取了 {len(room['wall_planes'])} 面墙")

    # 5. 提取门窗
    for semantic in annos["semantics"]:
        sem_type = semantic["type"]

        if sem_type == "door":
            geom = extract_opening_geometry(semantic, annos)
            if geom:
                cx, cy, cz, width, height = geom
                wall_ids = find_attached_walls(semantic, walls, annos)
                # 门通常挂靠在两面墙上（连接两个房间）
                for wall_id in wall_ids:
                    door = Door(
                        wall_id=wall_id, cx=cx, cy=cy, cz=cz, width=width, height=height
                    )
                    doors.append(door)

        elif sem_type == "window":
            geom = extract_opening_geometry(semantic, annos)
            if geom:
                cx, cy, cz, width, height = geom
                wall_ids = find_attached_walls(semantic, walls, annos)
                # 窗也可能挂靠在两面墙上
                for wall_id in wall_ids:
                    window = Window(
                        wall_id=wall_id, cx=cx, cy=cy, cz=cz, width=width, height=height
                    )
                    windows.append(window)

    print(f"\n总计: {len(walls)} 面墙, {len(doors)} 扇门, {len(windows)} 扇窗")

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
    print(f"\n结果已保存到 {filename}")


if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="从Structured3D格式提取墙体、门窗布局")
    parser.add_argument(
        "--input", default="annotation_3d.json", help="输入JSON文件路径"
    )
    parser.add_argument("--output", default="scene_output.txt", help="输出文件路径")
    parser.add_argument(
        "--room-labels", action="store_true", help="在输出中包含房间标注"
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 加载数据
    print("加载数据...")
    annos = load_json(args.input)

    # 提取布局
    print("提取布局信息...")
    walls, doors, windows = extract_layout(annos)

    # 格式化输出
    output = format_output(walls, doors, windows, include_room_labels=args.room_labels)

    # 打印结果
    if args.verbose:
        print("\n" + "=" * 50)
        print("输出结果:")
        print("=" * 50)
        print(output)

    # 显示统计信息
    print("\n统计信息:")
    print(f"  墙体数量: {len(walls)}")
    print(f"  门数量: {len(doors)}")
    print(f"  窗数量: {len(windows)}")

    # 按房间统计墙体
    room_wall_count = {}
    for wall in walls:
        if wall.room_type:
            if wall.room_type not in room_wall_count:
                room_wall_count[wall.room_type] = 0
            room_wall_count[wall.room_type] += 1

    print("\n按房间统计墙体:")
    for room_type, count in room_wall_count.items():
        print(f"  {room_type}: {count} 面墙")

    # 保存到文件
    save_output(output, args.output)
