import networkx as nx
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from spatiallm.layout.entity import Wall, Room
from spatiallm.layout.layout import Layout


def parse_walls(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    layout = Layout(content)
    # Convert list to dict for compatibility with existing logic
    # Layout.walls is a list of Wall objects
    walls = {}
    for wall in layout.walls:
        wall_id = f"wall_{wall.id}"
        walls[wall_id] = wall
    return walls


def build_graph(walls):
    G = nx.Graph()
    # Add nodes (walls)
    for w_id in walls:
        G.add_node(w_id)

    # Add edges if walls share a point (endpoint proximity)
    wall_ids = list(walls.keys())
    for i in range(len(wall_ids)):
        for j in range(i + 1, len(wall_ids)):
            w1_id = wall_ids[i]
            w2_id = wall_ids[j]
            w1 = walls[w1_id]
            w2 = walls[w2_id]
            w1 = walls[w1_id]
            w2 = walls[w2_id]

            # Check if endpoints are close
            # w1 endpoints: (ax, ay), (bx, by)
            # w2 endpoints: (ax, ay), (bx, by)
            # We only care about 2D projection (x, y) for connectivity

            threshold = 0.05  # 5cm tolerance

            p1_start = (w1.ax, w1.ay)
            p1_end = (w1.bx, w1.by)
            p2_start = (w2.ax, w2.ay)
            p2_end = (w2.bx, w2.by)

            connected = False
            for p1 in [p1_start, p1_end]:
                for p2 in [p2_start, p2_end]:
                    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                    if dist < threshold:
                        connected = True
                        break
                if connected:
                    break

            if connected:
                G.add_edge(w1_id, w2_id)

    return G


def find_rooms(walls):
    # This is a simplified approach. Finding rooms from walls is essentially finding cycles in the graph.
    # However, a simple cycle basis might not correspond exactly to rooms (e.g. nested rooms, or outer boundary).
    # For this task, we will find minimum cycles.

    G = build_graph(walls)
    cycles = nx.minimum_cycle_basis(G)

    rooms = []
    for i, cycle in enumerate(cycles):
        room_id = f"room_{i}"
        # Heuristic for room type: just generic "Room" for now as we can't infer semantic type from geometry easily without more context
        rooms.append(Room(id=room_id, wall_ids=cycle, type="Room"))

    return rooms


def sort_wall_ids(wall_ids):
    """Sort wall IDs numerically based on the number after 'wall_'"""

    def get_id_num(w_id):
        try:
            return int(w_id.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")

    return sorted(wall_ids, key=get_id_num)


def sort_walls_counter_clockwise(wall_ids, all_walls):
    """
    Sort walls in a counter-clockwise order based on connectivity and geometry.
    We traverse the walls based on connectivity to form a chain/cycle,
    and then check the signed area to ensure CCW orientation.
    """
    if not wall_ids:
        return []

    # Map wall_id -> Wall object
    room_walls = {wid: all_walls[wid] for wid in wall_ids if wid in all_walls}
    if not room_walls:
        return []

    w_ids = list(room_walls.keys())
    n = len(w_ids)
    if n < 3:
        # Too few walls to define a room area orientation reliably
        return w_ids

    # 1. Build adjacency for these walls
    adj = {wid: [] for wid in w_ids}
    threshold = 0.05  # 5cm tolerance

    # Pre-compute endpoints for efficiency
    endpoints = {}
    for wid, w in room_walls.items():
        endpoints[wid] = [(w.ax, w.ay), (w.bx, w.by)]

    for i in range(n):
        for j in range(i + 1, n):
            id1 = w_ids[i]
            id2 = w_ids[j]

            p1s = endpoints[id1]
            p2s = endpoints[id2]

            connected = False
            for p1 in p1s:
                for p2 in p2s:
                    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                    if dist < threshold:
                        connected = True
                        break
                if connected:
                    break

            if connected:
                adj[id1].append(id2)
                adj[id2].append(id1)

    # 2. Traverse graph to order walls
    # Start with a node with degree <= 2 (end of chain) or any node if cycle
    start_node = w_ids[0]
    for wid in w_ids:
        if len(adj[wid]) == 1:
            start_node = wid
            break

    path = [start_node]
    visited = {start_node}
    curr = start_node

    while len(path) < n:
        neighbors = adj[curr]
        next_node = None
        for neigh in neighbors:
            if neigh not in visited:
                next_node = neigh
                break

        if next_node is None:
            # If we are stuck (e.g. closed loop but not all walls visited, or disjoint)
            # If it is a proper room cycle, we should have visited everything or be at the start
            break

        path.append(next_node)
        visited.add(next_node)
        curr = next_node

    # Append any remaining walls (should not happen for valid single-room cycles)
    remaining = [w for w in w_ids if w not in visited]
    ordered_ids = path + remaining

    # 3. Determine Orientation (Signed Area)
    # Vertices are the intersection points between ordered walls
    vertices = []

    for i in range(len(ordered_ids)):
        w_curr_id = ordered_ids[i]
        w_next_id = ordered_ids[(i + 1) % len(ordered_ids)]

        p1s = endpoints[w_curr_id]
        p2s = endpoints[w_next_id]

        shared_pt = None

        for p1 in p1s:
            for p2 in p2s:
                dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                if dist < threshold:
                    shared_pt = p1
                    break
            if shared_pt:
                break

        if shared_pt:
            vertices.append(shared_pt)

    if len(vertices) < 3:
        return ordered_ids

    area = 0.0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    # Area > 0 implies CCW (assuming x right, y up standard Cartesian)
    # Area < 0 implies CW
    if area < 0:
        ordered_ids.reverse()

    return ordered_ids


def renumber_walls_in_layout(layout, rooms, all_walls):
    """
    Renumber walls sequentially based on room order.
    room_0's walls become wall_0, wall_1, ...
    room_1's walls become wall_N, wall_N+1, ...

    Returns:
        new_layout: Updated layout object
        new_rooms: Updated rooms list
    """

    # Map old_wall_id -> new_wall_id
    id_map = {}
    current_wall_idx = 0

    # Set to keep track of processed walls to avoid duplicates if walls are shared (unlikely in this simple model but good practice)
    processed_walls = set()

    # 1. Renumber walls in rooms
    for room in rooms:
        # Sort current room's walls counter-clockwise
        current_room_walls = sort_walls_counter_clockwise(room.wall_ids, all_walls)

        for old_id in current_room_walls:
            if old_id not in processed_walls:
                new_id_str = f"wall_{current_wall_idx}"
                id_map[old_id] = new_id_str
                processed_walls.add(old_id)
                current_wall_idx += 1

    # 2. Renumber remaining walls (if any) that are not part of any room
    for wall in layout.walls:
        old_id = f"wall_{wall.id}"
        if old_id not in processed_walls:
            new_id_str = f"wall_{current_wall_idx}"
            id_map[old_id] = new_id_str
            processed_walls.add(old_id)
            current_wall_idx += 1

    # Update Walls in Layout
    for wall in layout.walls:
        old_id = f"wall_{wall.id}"
        if old_id in id_map:
            new_id_num = int(id_map[old_id].split("_")[1])
            wall.id = new_id_num

    # Update Rooms
    for room in rooms:
        new_wall_ids = []
        # Re-sort using original IDs first to maintain the CCW order logic application
        # Actually, we should just map the sorted IDs we computed earlier?
        # The room.wall_ids were not updated in place in the loop above.

        # Let's re-compute the CCW sort on the original IDs to be consistent
        sorted_old_ids = sort_walls_counter_clockwise(room.wall_ids, all_walls)

        for old_w_id in sorted_old_ids:
            if old_w_id in id_map:
                new_wall_ids.append(id_map[old_w_id])
            else:
                new_wall_ids.append(old_w_id)

        # Now new_wall_ids contains the new IDs in CCW order
        room.wall_ids = new_wall_ids

    # Update Doors and Windows (they reference wall_id)
    # layout.doors and layout.windows use integer wall_id

    # Create integer map for easier lookup
    int_id_map = {}
    for old_str, new_str in id_map.items():
        old_num = int(old_str.split("_")[1])
        new_num = int(new_str.split("_")[1])
        int_id_map[old_num] = new_num

    for door in layout.doors:
        if door.wall_id in int_id_map:
            door.wall_id = int_id_map[door.wall_id]

    for window in layout.windows:
        if window.wall_id in int_id_map:
            window.wall_id = int_id_map[window.wall_id]

    return layout, rooms


def process_single_file(file_path, output_path):
    # print(f"Processing {file_path}...")
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Load everything into Layout
        layout = Layout(content)

        # Parse walls dict for graph building (needed for find_rooms)
        walls = {}
        for wall in layout.walls:
            wall_id = f"wall_{wall.id}"
            walls[wall_id] = wall

        # Find rooms based on geometry
        rooms = find_rooms(walls)

        # print(f"  Found {len(rooms)} rooms.")

        # Renumber walls sequentially
        layout, rooms = renumber_walls_in_layout(layout, rooms, walls)

        # Generate output
        lines = []

        # 1. Room lines
        for room in rooms:
            wall_ids_str = "[" + ",".join([f"'{w_id}'" for w_id in room.wall_ids]) + "]"
            line = f"{room.id}=Room({wall_ids_str},'{room.type}')"
            lines.append(line)

        # 2. Wall lines (sorted by new ID)
        # Sort walls by ID
        layout.walls.sort(key=lambda x: x.id)
        for wall in layout.walls:
            # wall_0=Wall(...)
            # We need to reconstruct the params string.
            # Wall definition: id, ax, ay, az, bx, by, bz, height, thickness
            line = f"wall_{wall.id}=Wall({wall.ax},{wall.ay},{wall.az},{wall.bx},{wall.by},{wall.bz},{wall.height},{wall.thickness})"
            lines.append(line)

        # 3. Door lines
        for door in layout.doors:
            # door_X=Door(wall_Y, pos_x, pos_y, pos_z, width, height)
            # Note: door.id is unique door ID, not related to wall numbering directly except via wall_id param
            # Check Layout.from_str for format:
            # door_0=Door(wall_1,1.5,0,0,0.9,2.1)
            line = f"door_{door.id}=Door(wall_{door.wall_id},{door.position_x},{door.position_y},{door.position_z},{door.width},{door.height})"
            lines.append(line)

        # 4. Window lines
        for window in layout.windows:
            # window_X=Window(wall_Y, pos_x, pos_y, pos_z, width, height)
            line = f"window_{window.id}=Window(wall_{window.wall_id},{window.position_x},{window.position_y},{window.position_z},{window.width},{window.height})"
            lines.append(line)

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        # print(f"  Saved to {output_path}")

    except Exception as e:
        tqdm.write(f"  Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Infer rooms and renumber walls for scene layout files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="test_data",
        help="Directory containing input scene txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_data_processed",
        help="Directory to save processed scene txt files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Find all txt files
    # Assuming files are named like scene_*.txt or just *.txt
    # We'll grab all .txt files
    files = list(input_dir.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process.")

    for file_path in tqdm(files, desc="Processing files"):
        filename = file_path.name
        # Avoid processing already processed files if input_dir == output_dir to prevent loops/overwrite issues if running multiple times?
        # But for now, simple logic.

        # If input is already "with_rooms", maybe skip or process again?
        # Let's assume we process everything.

        output_path = output_dir / filename
        process_single_file(str(file_path), str(output_path))


if __name__ == "__main__":
    main()
