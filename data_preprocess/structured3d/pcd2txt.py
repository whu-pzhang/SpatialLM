import json

import numpy as np

MM2M = 1.0 / 1000.0


def extract_walls(annos):
    planeLineMatrix = np.array(annos["planeLineMatrix"])
    junctions = np.array(annos["junctions"])
    lineJunctionMatrix = np.array(annos["lineJunctionMatrix"])

    # extract interior wall planes
    wall_planes = []
    # semantic type: living room, kitchen, bedroom, bathroom, etc.
    for semantic in annos["semantics"]:
        if semantic["type"] in ["door", "window", "outwall"]:
            continue
        for plane_id in semantic["planeID"]:
            if annos["planes"][plane_id]["type"] == "wall":
                wall_planes.append(plane_id)
    wall_planes = list(set(wall_planes))
    print(f"Total {len(wall_planes)} wall planes found.")

    doors = []
    windows = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["door", "window"]:
            for plane_id in semantic["planeID"]:
                # 组成当前plane的线段id
                line_indices = np.where(planeLineMatrix[plane_id])[0].tolist()
                print(line_indices)
                break
            break

    return wall_planes


def extract_doors_and_windows(annos):
    line_for_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                line_for_holes.append(
                    np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                )

    # 组成线段的点: [L, 2]
    # _, line_junction_indices = np.where(np.array(annos["lineJunctionMatrix"]))
    # line_junction_indices = line_junction_indices.reshape(-1, 2)
    line_junction_indices = np.stack(
        np.array(annos["lineJunctionMatrix"]).nonzero(), axis=1
    )

    junction_coords = np.array([j["coordinate"] for j in annos["junctions"]])

    walls = []
    for idx, plane in enumerate(planes):
        pid = plane["planeID"]
        line_ids = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[
            0
        ].tolist()
        wall_lines = line_junction_indices[line_ids]  # [num_lines, 2]
        wall_segments = junction_coords[wall_lines]  # [num_lines, 2, 3]

        walls.append(
            {
                "planeID": pid,
                "semantic_type": plane["type"],
                "segments_xyz": wall_segments,
                "line_ids": line_ids,
            }
        )
    return walls


if __name__ == "__main__":
    json_file = "data/Structured3D/scene_00000/annotation_3d.json"
    annos = json.load(open(json_file))
    walls = extract_walls(annos)
    print(walls)
    # print(walls[0])

    # layout = Layout("\n".join(walls))

    # pred_language_string = layout.to_language_string()

    # # check if the output path is a file or directory
    # with open("test_0000.txt", "w") as f:
    #     f.write(pred_language_string)
