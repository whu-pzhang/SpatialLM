import json

import numpy as np

# 加载annotation_3d.json文件
with open("data/Structured3D/scene_00000/annotation_3d.json", "r") as f:
    data = json.load(f)

# 提取数据
junctions = {j["ID"]: j["coordinate"] for j in data["junctions"]}
lines = data["lines"]
planes = {p["ID"]: p for p in data["planes"]}
semantics = data["semantics"]
plane_line_matrix = np.array(data["planeLineMatrix"])
line_junction_matrix = np.array(data["lineJunctionMatrix"])

# 找出所有的墙
walls = []
wall_planes = []
for semantic in semantics:
    if semantic["type"] == "wall":
        for plane_id in semantic["planeID"]:
            # 检查是否是垂直的墙面
            if abs(planes[plane_id]["normal"][2]) < 0.1:
                wall_planes.append(plane_id)

wall_planes = list(set(wall_planes))

for plane_id in wall_planes:
    # 找到属于这个平面的线
    line_indices = np.where(plane_line_matrix[plane_id] == 1)[0]
    for line_id in line_indices:
        # 找到线的两个端点
        junction_indices = np.where(line_junction_matrix[line_id] == 1)[0]
        if len(junction_indices) == 2:
            p1_id, p2_id = junction_indices
            p1 = junctions[p1_id]
            p2 = junctions[p2_id]

            # 我们只关心墙的底边
            if abs(p1[2] - p2[2]) < 0.01 and abs(p1[2]) < 0.01:
                # 假设z=0是地面，高度是2.8
                walls.append(Wall(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], 2.8))

# 找出所有的门窗
doors = []
windows = []
for semantic in semantics:
    if semantic["type"] in ["door", "window"]:
        for plane_id in semantic["planeID"]:
            # 找到属于这个平面的线
            line_indices = np.where(plane_line_matrix[plane_id] == 1)[0]

            # 计算开口的中心，宽度，和高度
            all_junctions_in_opening = []
            for line_id in line_indices:
                junction_indices = np.where(line_junction_matrix[line_id] == 1)[0]
                for j_id in junction_indices:
                    all_junctions_in_opening.append(junctions[j_id])

            if not all_junctions_in_opening:
                continue

            points = np.array(all_junctions_in_opening)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            center = (min_coords + max_coords) / 2
            width = np.linalg.norm(points[0] - points[1]) if len(points) > 1 else 0
            height = max_coords[2] - min_coords[2]

            # 找到门/窗所在的墙 (这里简化为找最近的墙)
            min_dist = float("inf")
            attached_wall_id = -1
            for i, wall in enumerate(walls):
                # 简化逻辑：找中点距离最近的墙
                wall_center = (
                    np.array([wall.a_x, wall.a_y, wall.a_z])
                    + np.array([wall.b_x, wall.b_y, wall.b_z])
                ) / 2
                dist = np.linalg.norm(center - wall_center)
                if dist < min_dist:
                    min_dist = dist
                    attached_wall_id = i

            if semantic["type"] == "door":
                doors.append(
                    Door(
                        f"wall_{attached_wall_id}",
                        center[0],
                        center[1],
                        center[2],
                        width,
                        height,
                    )
                )
            else:
                windows.append(
                    Window(
                        f"wall_{attached_wall_id}",
                        center[0],
                        center[1],
                        center[2],
                        width,
                        height,
                    )
                )
