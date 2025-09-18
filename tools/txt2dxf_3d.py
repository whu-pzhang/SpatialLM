"""
将SpatialLM预测的结果转换为 dxf 文件，方便后续进行评测和可视化。


1. 读取 SpatialLM 预测的文本文件，解析其中的几何信息: 可参考 spatiallm/layout
2. 将 Wall door 和 Windows 实体分别转换为三维的 DXF 实体


spatiallm 文本格式示例:

wall_k = Wall(ax, ay, az, bx, by, bz, height, thickness)
door_i   = Door(wall_id, cx, cy, cz, width, height)
window_j = Window(wall_id, cx, cy, cz, width, height)
"""

import math
import os
import sys

try:
    import ezdxf
    from ezdxf.document import Drawing
    from ezdxf.layouts import Modelspace

    from spatiallm.layout.layout import Door, Layout, Wall, Window

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"警告: 缺少必要的依赖库: {e}")
    print("请安装: pip install ezdxf")


class Txt2DxfConverter:
    """
    将 SpatialLM 文本文件转换为 DXF 文件的转换器。
    """

    def __init__(self, input_file: str, output_file: str):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("ezdxf 库不可用，无法执行转换")

        self.input_file = input_file
        self.output_file = output_file
        self.layout = self._parse_txt_file()

        self.doc: Drawing = ezdxf.new()
        self.msp: Modelspace = self.doc.modelspace()
        self._setup_layers()

    def _parse_txt_file(self) -> Layout:
        """解析 SpatialLM 预测的文本文件，提取几何信息"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"文件不存在: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            content = f.read()

        layout = Layout(content)
        return layout

    def _setup_layers(self):
        """设置 DXF 文件所需的图层"""
        self.doc.layers.new(name="Walls", dxfattribs={"color": 1})
        self.doc.layers.new(name="Doors", dxfattribs={"color": 3})
        self.doc.layers.new(name="Windows", dxfattribs={"color": 4})

    @staticmethod
    def _create_wall_vertices(wall: Wall) -> list:
        """为墙体创建线框顶点"""
        ax, ay, az = wall.ax, wall.ay, wall.az
        bx, by, bz = wall.bx, wall.by, wall.bz
        height = wall.height

        return [
            (ax, ay, az),
            (bx, by, bz),
            (bx, by, bz + height),
            (ax, ay, az + height),
        ]

    @staticmethod
    def _create_fixture_vertices(fixture: Door | Window, wall: Wall) -> list:
        """为门或窗创建线框顶点"""
        cx, cy, cz = fixture.position_x, fixture.position_y, fixture.position_z
        width = fixture.width
        height = fixture.height

        ax, ay = wall.ax, wall.ay
        bx, by = wall.bx, wall.by
        dx, dy = bx - ax, by - ay
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return []

        wall_dx, wall_dy = dx / length, dy / length
        half_w_dx = (width / 2) * wall_dx
        half_w_dy = (width / 2) * wall_dy

        z_bottom = cz - height / 2
        z_top = cz + height / 2

        return [
            (cx - half_w_dx, cy - half_w_dy, z_bottom),
            (cx + half_w_dx, cy + half_w_dy, z_bottom),
            (cx + half_w_dx, cy + half_w_dy, z_top),
            (cx - half_w_dx, cy - half_w_dy, z_top),
        ]

    def convert(self):
        """执行转换过程，生成 DXF 文件"""
        wall_map = {wall.id: wall for wall in self.layout.walls}

        # 绘制墙体
        for wall in self.layout.walls:
            vertices = self._create_wall_vertices(wall)
            if vertices:
                self.msp.add_polyline3d(
                    vertices, close=True, dxfattribs={"layer": "Walls"}
                )

        # 绘制门
        for door in self.layout.doors:
            wall = wall_map.get(door.wall_id)
            if wall:
                vertices = self._create_fixture_vertices(door, wall)
                if vertices:
                    self.msp.add_polyline3d(
                        vertices, close=True, dxfattribs={"layer": "Doors"}
                    )

        # 绘制窗
        for window in self.layout.windows:
            wall = wall_map.get(window.wall_id)
            if wall:
                vertices = self._create_fixture_vertices(window, wall)
                if vertices:
                    self.msp.add_polyline3d(
                        vertices, close=True, dxfattribs={"layer": "Windows"}
                    )

        self._save()

    def _save(self):
        """保存 DXF 文件"""
        self.doc.saveas(self.output_file)
        # print(f"DXF 文件已保存到: {self.output_file}")


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python txt2dxf_3d.py <输入文本文件> <输出DXF文件>")
        sys.exit(1)

    # 检查依赖
    if not DEPENDENCIES_AVAILABLE:
        print("\n错误: 缺少必要的依赖库。请先安装 ezdxf。")
        print("命令: pip install ezdxf")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        converter = Txt2DxfConverter(input_file, output_file)
        converter.convert()
        return 0
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"\n错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
