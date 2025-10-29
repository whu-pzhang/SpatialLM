"""
Converts the prediction results of SpatialLM into a DXF file for subsequent
evaluation and visualization.


1. Read the text file predicted by SpatialLM and parse the geometric information:
   refer to spatiallm/layout
2. Convert Wall, Door, and Window entities into 3D DXF entities.


Example of spatiallm text format:

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
    print(f"Warning: Missing required dependencies: {e}")
    print("Please install: pip install ezdxf")


class Txt2DxfConverter:
    """
    Converter for transforming SpatialLM text files into DXF files.
    """

    def __init__(self, input_file: str, output_file: str):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "ezdxf library is not available, cannot perform conversion"
            )

        self.input_file = input_file
        self.output_file = output_file
        self.layout = self._parse_txt_file()

        self.doc: Drawing = ezdxf.new()
        self.msp: Modelspace = self.doc.modelspace()
        self._setup_layers()

    def _parse_txt_file(self) -> Layout:
        """Parse the SpatialLM prediction text file to extract geometric info."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"File not found: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            content = f.read()

        layout = Layout(content)
        return layout

    def _setup_layers(self):
        """Set up the necessary layers for the DXF file."""
        self.doc.layers.new(name="Walls", dxfattribs={"color": 1})
        self.doc.layers.new(name="Doors", dxfattribs={"color": 3})
        self.doc.layers.new(name="Windows", dxfattribs={"color": 4})

    @staticmethod
    def _create_wall_vertices(wall: Wall) -> list:
        """Create wireframe vertices for a wall."""
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
        """Create wireframe vertices for a door or window."""
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
        """Execute the conversion process to generate the DXF file."""
        wall_map = {wall.id: wall for wall in self.layout.walls}

        # Draw walls
        for wall in self.layout.walls:
            vertices = self._create_wall_vertices(wall)
            if vertices:
                self.msp.add_polyline3d(
                    vertices, close=True, dxfattribs={"layer": "Walls"}
                )

        # Draw doors
        for door in self.layout.doors:
            wall = wall_map.get(door.wall_id)
            if wall:
                vertices = self._create_fixture_vertices(door, wall)
                if vertices:
                    self.msp.add_polyline3d(
                        vertices, close=True, dxfattribs={"layer": "Doors"}
                    )

        # Draw windows
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
        """Save the DXF file."""
        self.doc.saveas(self.output_file)
        # print(f"DXF file saved to: {self.output_file}")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python txt2dxf_3d.py <input_text_file> <output_dxf_file>")
        sys.exit(1)

    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("\nError: Missing required dependencies. Please install ezdxf first.")
        print("Command: pip install ezdxf")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        converter = Txt2DxfConverter(input_file, output_file)
        converter.convert()
        return 0
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
