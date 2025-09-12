import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from spatiallm import Layout
from spatiallm.pcd import get_points_and_colors, load_o3d_pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout Visualization with rerun")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        required=True,
        help="Path to the layout txt file",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.01,
        help="The radius of the points for visualization",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=2000000,
        help="The maximum number of points for visualization",
    )
    parser.add_argument(
        "--hide-labels", action="store_true", help="Hide labels in the visualization"
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    save_dir = Path(args.save).parent
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.layout, "r") as f:
        layout_content = f.read()

    pcd = load_o3d_pcd(args.point_cloud)
    points, colors = get_points_and_colors(pcd)

    # parse layout_content
    layout = Layout(layout_content)
    floor_plan = layout.to_boxes()

    # ReRun visualization
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        rrb.Spatial2DView(
            name="Floor Plan", origin="/world", background=[255, 255, 255]
        ),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    point_indices = np.arange(points.shape[0])
    np.random.shuffle(point_indices)
    point_indices = point_indices[: args.max_points]
    points = points[point_indices]
    colors = colors[point_indices]
    rr.log(
        "world/points",
        rr.Points3D(
            positions=points,
            colors=colors,
            radii=args.radius,
        ),
        static=True,
    )

    num_entities = len(floor_plan)
    seconds = 0.5
    for ti in range(num_entities + 1):
        sub_floor_plan = floor_plan[:ti]

        rr.set_time_seconds("time_sec", ti * seconds)
        for box in sub_floor_plan:
            uid = box["id"]
            group = box["class"]
            label = None if args.hide_labels else box["label"]

            rr.log(
                f"world/pred/{group}/{uid}",
                rr.Boxes3D(
                    centers=box["center"],
                    half_sizes=0.5 * box["scale"],
                    labels=label,
                ),
                rr.InstancePoses3D(mat3x3=box["rotation"]),
            )

            # Draw the 2D floor plan as wall lines
            # scale = [length, thickness, height], thickness为零，绘制为线条
            center_3d = box["center"]
            scale = box["scale"]
            rotation_3d = box["rotation"]

            # length是墙的长度，thickness为零
            half_length = 0.5 * scale[0]

            # 计算墙体两端点（沿着墙的长度方向）
            # 在局部坐标系中，墙沿X轴方向
            local_endpoints = np.array(
                [
                    [-half_length, 0, 0],  # 起点
                    [half_length, 0, 0],  # 终点
                ]
            )

            # 应用3D旋转变换
            world_endpoints = (rotation_3d @ local_endpoints.T).T + center_3d

            # 取XY坐标作为2D线条
            transformed_corners = world_endpoints[:, :2]

            rr.log(
                f"world/floor_plan/{group}/{uid}",
                rr.LineStrips2D([transformed_corners]),
            )
            if label and not args.hide_labels:
                # Log a point at the center of the box to display the label,
                # which is a robust way to add text in space in older rerun versions.
                # 标签位置使用中心点
                # 标签位置使用中心点的XY坐标
                label_center = center_3d[:2]
                rr.log(
                    f"world/floor_plan/{group}/{uid}/label",
                    rr.Points2D(positions=[label_center], labels=[label]),
                )
    rr.script_teardown(args)
