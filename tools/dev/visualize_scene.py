import matplotlib.pyplot as plt
import numpy as np
import argparse
from spatiallm.layout.layout import Layout


def parse_scene_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    layout = Layout(content)

    # Convert lists to dicts for easier access
    walls = {w.id: w for w in layout.walls}
    rooms = {r.id: r for r in layout.rooms}
    doors = {d.id: d for d in layout.doors}
    windows = {w.id: w for w in layout.windows}

    return walls, rooms, doors, windows


def visualize_scene(
    walls, rooms, doors, windows, output_path="scene_visualization.png"
):
    plt.figure(figsize=(12, 12))

    # Define colors for different rooms
    colors = [
        "#FF9999",
        "#66B2FF",
        "#99FF99",
        "#FFCC99",
        "#FFD700",
        "#FF69B4",
        "#8A2BE2",
        "#00CED1",
    ]

    # Pre-compute room centroids and wall-to-room mapping
    room_centroids = {}
    wall_to_rooms = {}

    for room_id, room in rooms.items():
        room_walls = []
        for w_id_str in room.wall_ids:
            try:
                w_id = int(w_id_str.split("_")[1])
                if w_id in walls:
                    room_walls.append(walls[w_id])
                    if w_id not in wall_to_rooms:
                        wall_to_rooms[w_id] = []
                    wall_to_rooms[w_id].append(room_id)
            except (ValueError, IndexError):
                continue

        if room_walls:
            cx, cy = 0, 0
            for w in room_walls:
                cx += w.ax + w.bx
                cy += w.ay + w.by
            cx /= 2 * len(room_walls)
            cy /= 2 * len(room_walls)
            room_centroids[room_id] = (cx, cy)

    # Plot rooms (polygons formed by walls)
    for i, (room_id, room) in enumerate(rooms.items()):
        color = colors[i % len(colors)]

        # Collect wall segments again for plotting
        room_walls = []
        for w_id_str in room.wall_ids:
            try:
                w_id = int(w_id_str.split("_")[1])
                if w_id in walls:
                    room_walls.append(walls[w_id])
            except:
                pass

        if not room_walls:
            continue

        for wall in room_walls:
            plt.plot(
                [wall.ax, wall.bx],
                [wall.ay, wall.by],
                color=color,
                linewidth=6,
                alpha=0.6,
                label=f"{room_id}" if i == 0 else "",
            )

            # Plot wall ID at center of wall
            mx = (wall.ax + wall.bx) / 2
            my = (wall.ay + wall.by) / 2
            plt.text(
                mx,
                my,
                str(wall.id),
                fontsize=8,
                color="black",
                ha="center",
                va="center",
            )

        if room_id in room_centroids:
            cx, cy = room_centroids[room_id]
            plt.text(
                cx,
                cy,
                room_id,
                fontsize=12,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    # Plot all walls (structure)
    for w_id, wall in walls.items():
        # Check if wall is not already drawn (part of a room)
        in_room = False
        for room in rooms.values():
            if f"wall_{w_id}" in room.wall_ids:
                in_room = True
                break
        if not in_room:
            plt.plot(
                [wall.ax, wall.bx],
                [wall.ay, wall.by],
                "k-",
                linewidth=1,
                alpha=0.3,
                zorder=-1,
            )
            # Label loose walls
            mx = (wall.ax + wall.bx) / 2
            my = (wall.ay + wall.by) / 2
            plt.text(
                mx, my, str(wall.id), fontsize=6, color="gray", ha="center", va="center"
            )

    # Helper function to plot openings
    def plot_opening(opening, color, marker, label_prefix):
        if opening.wall_id in walls:
            wall = walls[opening.wall_id]
            # Calculate direction vector of the wall
            dx = wall.bx - wall.ax
            dy = wall.by - wall.ay
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                ux = dx / length
                uy = dy / length

                # Opening center
                cx, cy = opening.position_x, opening.position_y

                # Start and end points of the opening
                half_width = opening.width / 2
                sx = cx - ux * half_width
                sy = cy - uy * half_width
                ex = cx + ux * half_width
                ey = cy + uy * half_width

                # Plot thick line for opening
                plt.plot([sx, ex], [sy, ey], color=color, linewidth=4, zorder=10)
                # Plot marker at center
                plt.plot(
                    cx,
                    cy,
                    marker,
                    color="white",
                    markeredgecolor=color,
                    markersize=8,
                    zorder=11,
                )

                # Label with offset to avoid overlapping with the marker
                # Calculate normal vector (rotated 90 degrees from wall direction)
                n1x, n1y = -uy, ux
                n2x, n2y = uy, -ux

                # Default normal
                nx, ny = n1x, n1y

                # Determine which normal points towards the room center
                if opening.wall_id in wall_to_rooms:
                    # Use the first room this wall belongs to (usually shared walls belong to 2 rooms, pick one)
                    # Or ideally, if shared, maybe label twice? For now, pick first.
                    room_id = wall_to_rooms[opening.wall_id][0]
                    if room_id in room_centroids:
                        rcx, rcy = room_centroids[room_id]
                        # Vector from opening to room centroid
                        vx = rcx - cx
                        vy = rcy - cy

                        # Dot product to check alignment
                        dot1 = n1x * vx + n1y * vy
                        dot2 = n2x * vx + n2y * vy

                        if dot2 > dot1:
                            nx, ny = n2x, n2y

                # Apply a small offset perpendicular to the wall
                offset_dist = 0.2
                lx = cx + nx * offset_dist
                ly = cy + ny * offset_dist

                plt.text(
                    lx,
                    ly,
                    f"{label_prefix}{opening.id}",
                    fontsize=10,
                    color=color,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.9,
                        edgecolor=color,
                        boxstyle="round,pad=0.2",
                    ),
                    zorder=15,
                )

    # Plot Doors
    for door in doors.values():
        plot_opening(door, "#8B4513", "o", "D")  # SaddleBrown for doors

    # Plot Windows
    for window in windows.values():
        plot_opening(window, "#00008B", "s", "W")  # DarkBlue for windows

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Renumbered Scene Visualization (Rooms, Walls, Doors, Windows)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")

    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a scene layout from a text file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="test_data/scene_00000_with_rooms.txt",
        help="Path to the input scene text file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scene_renumbered_visualization.png",
        help="Path to save the visualization image.",
    )

    args = parser.parse_args()

    file_path = args.input_path
    walls, rooms, doors, windows = parse_scene_file(file_path)
    print(
        f"Loaded {len(walls)} walls, {len(rooms)} rooms, {len(doors)} doors, {len(windows)} windows."
    )
    visualize_scene(walls, rooms, doors, windows, output_path=args.output)
