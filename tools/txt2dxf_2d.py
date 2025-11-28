"""
Converts the prediction results of SpatialLM into a DXF file for subsequent
evaluation and visualization.


1. Read the text file predicted by SpatialLM and parse the geometric information:
   refer to spatiallm/layout
2. Keep only the floor plan of the Wall structure, ignoring 3D information
   such as height.
3. Combine the planar line segments that make up the Wall into multiple closed
   polygons by room.
4. Convert each closed polygon into an LWPOLYLINE entity in the DXF file.
5. Write the polygons of all rooms to the same DXF layer, named "wall_poly".

Example of spatiallm text format:

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

# Add the project root directory to the path to import the spatiallm module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ezdxf
    from spatiallm.layout.layout import Layout

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Missing required dependencies: {e}")
    print("Please install: pip install ezdxf")


class WallSegment:
    """Represents a wall segment."""

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
    """Connected component analyzer to group connected wall segments."""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def are_points_connected(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """Check if two points are equal within a tolerance."""
        return np.linalg.norm(np.array(p1) - np.array(p2)) < self.tolerance

    def find_connected_components(
        self, wall_segments: List[WallSegment]
    ) -> List[List[WallSegment]]:
        """Find all connected components using a Disjoint Set Union (DSU) algorithm."""
        if not wall_segments:
            return []

        # Build a map from points to walls
        point_to_walls = defaultdict(list)
        for wall in wall_segments:
            point_to_walls[wall.start_point].append(wall)
            point_to_walls[wall.end_point].append(wall)

        # Merge nearby points
        merged_points = self._merge_nearby_points(list(point_to_walls.keys()))

        # Rebuild the map from points to walls
        point_to_walls_merged = defaultdict(list)
        for wall in wall_segments:
            start_merged = self._find_merged_point(wall.start_point, merged_points)
            end_merged = self._find_merged_point(wall.end_point, merged_points)
            point_to_walls_merged[start_merged].append(wall)
            point_to_walls_merged[end_merged].append(wall)

        # Find connected components using DSU
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

        # Union walls that share an endpoint
        for point, walls in point_to_walls_merged.items():
            if len(walls) > 1:
                for i in range(1, len(walls)):
                    union(walls[0].wall_id, walls[i].wall_id)

        # Group walls by connected component
        components = defaultdict(list)
        for wall in wall_segments:
            root = find(wall.wall_id)
            components[root].append(wall)

        return list(components.values())

    def _merge_nearby_points(
        self, points: List[Tuple[float, float]]
    ) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """Merge nearby points and return a map from original to merged points."""
        merged = {}
        representatives = []

        for point in points:
            # Find the closest representative point
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
        """Find the merged point corresponding to a given point."""
        return merged_points.get(point, point)


class PolygonBuilder:
    """Builds polygons from connected wall segments."""

    def __init__(self, tolerance: float = 1e-3, keep_open_walls: bool = False):
        self.tolerance = tolerance
        self.keep_open_walls = keep_open_walls

    def build_polygons(
        self, wall_segments: List[WallSegment]
    ) -> List[List[Tuple[float, float]]]:
        """Connect wall segments into closed polygons."""
        if not wall_segments:
            return []

        # Build adjacency graph
        adjacency = self._build_adjacency_graph(wall_segments)

        # Find all possible closed paths
        polygons = []
        used_walls = set()

        for wall in wall_segments:
            if wall.wall_id in used_walls:
                continue

            polygons_from_wall = self._find_polygons(
                wall, adjacency, used_walls, wall_segments
            )
            polygons.extend(polygons_from_wall)

        # If keep_open_walls is True, add open wall segments as polylines
        if self.keep_open_walls:
            open_walls = [
                wall for wall in wall_segments if wall.wall_id not in used_walls
            ]
            for wall in open_walls:
                # Add open wall as a simple line segment (2-point polygon)
                open_poly = [wall.start_point, wall.end_point]
                polygons.append(open_poly)

        return polygons

    def _build_adjacency_graph(
        self, wall_segments: List[WallSegment]
    ) -> Dict[Tuple[float, float], List[WallSegment]]:
        """Build an adjacency graph based on endpoints."""
        adjacency = defaultdict(list)
        for wall in wall_segments:
            adjacency[wall.start_point].append(wall)
            adjacency[wall.end_point].append(wall)
        return adjacency

    def _are_points_connected(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """Check if two points are connected."""
        return np.linalg.norm(np.array(p1) - np.array(p2)) < self.tolerance

    def _find_polygons(
        self,
        start_wall: WallSegment,
        adjacency: Dict[Tuple[float, float], List[WallSegment]],
        used_walls: Set[int],
        wall_segments: List[WallSegment],
    ) -> List[List[Tuple[float, float]]]:
        """Find all closed polygons from a connected component."""
        if start_wall.wall_id in used_walls:
            return []

        polygons = []

        # `traverse` is a helper function to perform depth-first search
        def traverse(
            current_wall: WallSegment,
            path: List[Tuple[float, float]],
            visited_in_path: Set[int],
        ):
            nonlocal polygons

            # Add the current wall to the visited set for the current path
            visited_in_path.add(current_wall.wall_id)

            # Find the next connected wall
            last_point = path[-1]

            # Find walls connected to the end of the current path
            connected_walls = [
                w for w in adjacency[last_point] if w.wall_id != current_wall.wall_id
            ]

            for next_wall in connected_walls:
                # If the next wall is the starting wall, a polygon is formed
                if next_wall.wall_id == start_wall.wall_id:
                    if len(path) > 2:
                        polygons.append(path)
                    continue

                # If the wall is already in the current path, skip it
                if next_wall.wall_id in visited_in_path:
                    continue

                # Otherwise, continue traversing
                next_point = (
                    next_wall.start_point
                    if self._are_points_connected(last_point, next_wall.end_point)
                    else next_wall.end_point
                )
                traverse(next_wall, path + [next_point], visited_in_path.copy())

        # Start traversal from the starting wall
        initial_path = [start_wall.start_point, start_wall.end_point]
        traverse(start_wall, initial_path, {start_wall.wall_id})

        # Mark all walls in the found polygons as used
        for poly in polygons:
            for wall in wall_segments:
                if any(
                    self._are_points_connected(wall.start_point, p) for p in poly
                ) and any(self._are_points_connected(wall.end_point, p) for p in poly):
                    used_walls.add(wall.wall_id)

        return polygons


class DXFGenerator:
    """DXF file generator."""

    def __init__(self, layer_name: str = "wall_poly"):
        self.doc = None
        self.msp = None
        self.layer_name = layer_name

    def create_dxf(self, polygons: List[List[Tuple[float, float]]], output_file: str):
        """Create a DXF file."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "ezdxf library is not available, cannot generate DXF file"
            )

        # Create a new DXF document
        self.doc = ezdxf.new("R2010")
        self.msp = self.doc.modelspace()

        # Create a new layer
        self.doc.layers.new(name=self.layer_name, dxfattribs={"color": 7})

        # Add polygons to the DXF
        for i, polygon in enumerate(polygons):
            if len(polygon) >= 3:
                self._add_polygon_to_dxf(polygon, self.layer_name)
            elif len(polygon) == 2:
                # Handle open wall segments (line segments)
                self._add_line_to_dxf(polygon, self.layer_name)

        # Save the DXF file
        self.doc.saveas(output_file)
        # print(f"DXF 文件已保存到: {output_file}")

    def _add_polygon_to_dxf(self, polygon: List[Tuple[float, float]], layer_name: str):
        """Add a polygon to the DXF file."""
        # Ensure the polygon is closed
        points = polygon.copy()
        if len(points) > 2 and points[0] != points[-1]:
            points.append(points[0])

        # Create an LWPOLYLINE entity
        lwpolyline = self.msp.add_lwpolyline(points)
        lwpolyline.dxf.layer = layer_name
        lwpolyline.closed = True

    def _add_line_to_dxf(self, line: List[Tuple[float, float]], layer_name: str):
        """Add a line segment (open wall) to the DXF file."""
        # Create a LINE entity for open wall segments
        line_entity = self.msp.add_line(line[0], line[1])
        line_entity.dxf.layer = layer_name


def parse_spatiallm_text(text_file: str) -> List[WallSegment]:
    """Parse a SpatialLM text file and extract wall information."""
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"File not found: {text_file}")

    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Use the existing Layout class for parsing
    layout = Layout(content)

    # Extract wall segments (using only X, Y coordinates, ignoring Z)
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
    keep_open_walls=False,
):
    wall_segments = parse_spatiallm_text(input_file)

    if not wall_segments:
        print("Warning: No wall information found")
        return 0

    analyzer = ConnectedComponentAnalyzer(tolerance=1e-3)
    components = analyzer.find_connected_components(wall_segments)

    all_polygons = []
    builder = PolygonBuilder(tolerance=1e-3, keep_open_walls=keep_open_walls)

    for i, component in enumerate(components):
        if verbose:
            print(
                f"Processing connected component {i + 1}/{len(components)} ({len(component)} walls)"
            )

        polygons = builder.build_polygons(component)
        all_polygons.extend(polygons)

    generator = DXFGenerator(layer_name=layer_name)
    generator.create_dxf(all_polygons, output_file)

    if verbose:
        print(f"Conversion complete! Output file: {output_file}")
        print(f"- Processed {len(wall_segments)} wall segments")
        print(f"- Identified {len(components)} connected components")
        print(f"- Generated {len(all_polygons)} polygons")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert SpatialLM prediction results to DXF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python txt2dxf.py scene_00000.txt -o output.dxf
  python txt2dxf.py input.txt --tolerance 0.001
        """,
    )

    parser.add_argument("input_file", help="Input SpatialLM text file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output DXF file path (defaults to input filename.dxf)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="wall_poly",
        help="DXF layer name (default: wall_poly)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Point connection tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    parser.add_argument(
        "--keep-open-walls",
        action="store_true",
        help="Keep open walls (non-closed wall segments) in the output",
    )

    args = parser.parse_args()

    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Missing required dependencies")
        print("Please install: pip install ezdxf")
        return 1

    # Determine output filename
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
            keep_open_walls=args.keep_open_walls,
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
