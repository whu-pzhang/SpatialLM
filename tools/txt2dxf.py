"""
将SpatialLM预测的结果转换为 dxf 文件，方便后续进行评测和可视化。


1. 读取 SpatialLM 预测的文本文件，解析其中的几何信息: 可参考 spatiallm/layout
2. 只保留 Wall 结构的平面图，忽略高度信息等三维信息
3. 将组成 Wall 的平面线段按房间组合为多个闭合多边形
4. 将每个闭合多边形转换为 DXF 文件中的 LWPOLYLINE 实体
5. 将所有房间的多边形写入同一个 DXF 图层，图层名为 "wall_poly"

spatiallm 文本格式示例:

wall_k = Wall(ax, ay, az, bx, by, bz, height, thickness)
door_i   = Door(wall_id, cx, cy, cz, width, height)
window_j = Window(wall_id, cx, cy, cz, width, height)
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

# 添加项目根目录到路径，以便导入 spatiallm 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ezdxf

    from spatiallm.layout.layout import Layout

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"警告: 缺少必要的依赖库: {e}")
    print("请安装: pip install ezdxf")


class WallSegment:
    """表示一个墙体线段"""

    def __init__(
        self,
        wall_id: int,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
    ):
        self.wall_id = wall_id
        self.start_point = start_point
        self.end_point = end_point

    def __repr__(self):
        return f"Wall_{self.wall_id}: {self.start_point} -> {self.end_point}"


class ConnectedComponentAnalyzer:
    """连通分量分析器，用于将连通的墙体线段分组"""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def are_points_connected(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """判断两个点是否在容差范围内相等"""
        return np.linalg.norm(np.array(p1) - np.array(p2)) < self.tolerance

    def find_connected_components(
        self, wall_segments: List[WallSegment]
    ) -> List[List[WallSegment]]:
        """使用并查集算法找出所有连通分量"""
        if not wall_segments:
            return []

        # 构建点到墙体的映射
        point_to_walls = defaultdict(list)
        for wall in wall_segments:
            point_to_walls[wall.start_point].append(wall)
            point_to_walls[wall.end_point].append(wall)

        # 合并相近的点
        merged_points = self._merge_nearby_points(list(point_to_walls.keys()))

        # 重新构建点到墙体的映射
        point_to_walls_merged = defaultdict(list)
        for wall in wall_segments:
            start_merged = self._find_merged_point(wall.start_point, merged_points)
            end_merged = self._find_merged_point(wall.end_point, merged_points)
            point_to_walls_merged[start_merged].append(wall)
            point_to_walls_merged[end_merged].append(wall)

        # 使用并查集找连通分量
        parent = {wall.wall_id: wall.wall_id for wall in wall_segments}

        def find(wall_id):
            if parent[wall_id] == wall_id:
                return wall_id
            parent[wall_id] = find(parent[wall_id])
            return parent[wall_id]

        def union(wall_id1, wall_id2):
            root1 = find(wall_id1)
            root2 = find(wall_id2)
            if root1 != root2:
                parent[root2] = root1

        # 将共享端点的墙体合并到同一连通分量
        for point, walls in point_to_walls_merged.items():
            if len(walls) > 1:
                for i in range(1, len(walls)):
                    union(walls[0].wall_id, walls[i].wall_id)

        # 按连通分量分组
        components = defaultdict(list)
        for wall in wall_segments:
            root = find(wall.wall_id)
            components[root].append(wall)

        return list(components.values())

    def _merge_nearby_points(
        self, points: List[Tuple[float, float]]
    ) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """合并相近的点，返回原点到合并后点的映射"""
        merged = {}
        representatives = []

        for point in points:
            # 找到最近的代表点
            closest_rep = None
            min_dist = float("inf")

            for rep in representatives:
                dist = np.linalg.norm(np.array(point) - np.array(rep))
                if dist < min_dist:
                    min_dist = dist
                    closest_rep = rep

            if closest_rep is not None and min_dist < self.tolerance:
                merged[point] = closest_rep
            else:
                representatives.append(point)
                merged[point] = point

        return merged

    def _find_merged_point(
        self,
        point: Tuple[float, float],
        merged_points: Dict[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[float, float]:
        """找到点对应的合并后的点"""
        return merged_points.get(point, point)


class PolygonBuilder:
    """多边形构建器，将连通的墙体线段连接成闭合多边形"""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def build_polygons(
        self, wall_segments: List[WallSegment]
    ) -> List[List[Tuple[float, float]]]:
        """将墙体线段连接成闭合多边形"""
        if not wall_segments:
            return []

        # 构建邻接图
        adjacency = self._build_adjacency_graph(wall_segments)

        # 寻找所有可能的闭合路径
        polygons = []
        used_walls = set()

        for wall in wall_segments:
            if wall.wall_id in used_walls:
                continue

            polygons_from_wall = self._find_polygons(
                wall, adjacency, used_walls, wall_segments
            )
            polygons.extend(polygons_from_wall)

        return polygons

    def _build_adjacency_graph(
        self, wall_segments: List[WallSegment]
    ) -> Dict[Tuple[float, float], List[WallSegment]]:
        """构建基于端点的邻接图"""
        adjacency = defaultdict(list)
        for wall in wall_segments:
            adjacency[wall.start_point].append(wall)
            adjacency[wall.end_point].append(wall)
        return adjacency

    def _are_points_connected(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """判断两个点是否连接"""
        return np.linalg.norm(np.array(p1) - np.array(p2)) < self.tolerance

    def _find_polygons(
        self,
        start_wall: WallSegment,
        adjacency: Dict[Tuple[float, float], List[WallSegment]],
        used_walls: Set[int],
        wall_segments: List[WallSegment],
    ) -> List[List[Tuple[float, float]]]:
        """从一个连通分量中找出所有闭合多边形"""
        if start_wall.wall_id in used_walls:
            return []

        polygons = []

        # `traverse` 是一个辅助函数，用于执行深度优先搜索
        def traverse(
            current_wall: WallSegment,
            path: List[Tuple[float, float]],
            visited_in_path: Set[int],
        ):
            nonlocal polygons

            # 将当前墙体添加到已访问集合
            visited_in_path.add(current_wall.wall_id)

            # 查找下一个连接的墙体
            last_point = path[-1]

            # 找到与当前路径末端连接的墙体
            connected_walls = [
                w for w in adjacency[last_point] if w.wall_id != current_wall.wall_id
            ]

            for next_wall in connected_walls:
                # 如果下一个墙体是路径的起点，则形成闭合
                if next_wall.wall_id == start_wall.wall_id:
                    if len(path) > 2:
                        polygons.append(path)
                    continue

                # 如果墙体已在当前路径中，跳过
                if next_wall.wall_id in visited_in_path:
                    continue

                # 否则，继续遍历
                next_point = (
                    next_wall.start_point
                    if self._are_points_connected(last_point, next_wall.end_point)
                    else next_wall.end_point
                )
                traverse(next_wall, path + [next_point], visited_in_path.copy())

        # 从起始墙体开始遍历
        initial_path = [start_wall.start_point, start_wall.end_point]
        traverse(start_wall, initial_path, {start_wall.wall_id})

        # 标记所有找到的多边形中的墙体为已使用
        for poly in polygons:
            for wall in wall_segments:
                if any(
                    self._are_points_connected(wall.start_point, p) for p in poly
                ) and any(self._are_points_connected(wall.end_point, p) for p in poly):
                    used_walls.add(wall.wall_id)

        return polygons


class DXFGenerator:
    """DXF文件生成器"""

    def __init__(self, layer_name: str = "wall_poly"):
        self.doc = None
        self.msp = None
        self.layer_name = layer_name

    def create_dxf(self, polygons: List[List[Tuple[float, float]]], output_file: str):
        """创建DXF文件"""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("ezdxf 库不可用，无法生成 DXF 文件")

        # 创建新的DXF文档
        self.doc = ezdxf.new("R2010")
        self.msp = self.doc.modelspace()

        # 创建图层
        self.doc.layers.new(name=self.layer_name, dxfattribs={"color": 7})

        # 添加多边形到DXF
        for i, polygon in enumerate(polygons):
            if len(polygon) >= 3:
                self._add_polygon_to_dxf(polygon, self.layer_name)

        # 保存文件
        self.doc.saveas(output_file)
        # print(f"DXF 文件已保存到: {output_file}")

    def _add_polygon_to_dxf(self, polygon: List[Tuple[float, float]], layer_name: str):
        """将多边形添加到DXF文件"""
        # 确保多边形是闭合的
        points = polygon.copy()
        if len(points) > 2 and points[0] != points[-1]:
            points.append(points[0])

        # 创建LWPOLYLINE实体
        lwpolyline = self.msp.add_lwpolyline(points)
        lwpolyline.dxf.layer = layer_name
        lwpolyline.closed = True


def parse_spatiallm_text(text_file: str) -> List[WallSegment]:
    """解析SpatialLM文本文件，提取墙体信息"""
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"文件不存在: {text_file}")

    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 使用现有的Layout类解析
    layout = Layout(content)

    # 提取墙体线段（只使用X,Y坐标，忽略Z坐标）
    wall_segments = []
    for wall in layout.walls:
        start_point = (wall.ax, wall.ay)
        end_point = (wall.bx, wall.by)
        wall_segments.append(WallSegment(wall.id, start_point, end_point))

    return wall_segments


def txt2dxf(
    input_file: str,
    output_file: str,
    layer_name: str = "wall_poly",
    verbose=False,
):
    wall_segments = parse_spatiallm_text(input_file)

    if not wall_segments:
        print("警告: 未找到任何墙体信息")
        return 0

    analyzer = ConnectedComponentAnalyzer(tolerance=1e-3)
    components = analyzer.find_connected_components(wall_segments)

    all_polygons = []
    builder = PolygonBuilder(tolerance=1e-3)

    for i, component in enumerate(components):
        if verbose:
            print(f"处理连通分量 {i + 1}/{len(components)} ({len(component)} 个墙体)")

        polygons = builder.build_polygons(component)
        all_polygons.extend(polygons)

    generator = DXFGenerator(layer_name=layer_name)
    generator.create_dxf(all_polygons, output_file)

    if verbose:
        print(f"转换完成! 输出文件: {output_file}")
        print(f"- 处理了 {len(wall_segments)} 个墙体线段")
        print(f"- 识别了 {len(components)} 个连通分量")
        print(f"- 生成了 {len(all_polygons)} 个多边形")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将 SpatialLM 预测结果转换为 DXF 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python txt2dxf.py scene_00000.txt -o output.dxf
  python txt2dxf.py input.txt --tolerance 0.001
        """,
    )

    parser.add_argument("input_file", help="输入的 SpatialLM 文本文件")
    parser.add_argument(
        "-o", "--output", help="输出的 DXF 文件路径（默认为输入文件名.dxf）"
    )
    parser.add_argument(
        "--layer", type=str, default="wall_poly", help="DXF 图层名（默认: wall_poly）"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3, help="点连接容差（默认: 1e-3）"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 检查依赖
    if not DEPENDENCIES_AVAILABLE:
        print("错误: 缺少必要的依赖库")
        print("请安装: pip install ezdxf")
        return 1

    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}.dxf"

    try:
        txt2dxf(
            input_file=args.input_file,
            output_file=output_file,
            layer_name=args.layer,
            verbose=args.verbose,
        )

        return 0

    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
