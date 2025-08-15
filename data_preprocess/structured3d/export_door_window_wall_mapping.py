#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Structured3D 风格的 annotation_3d.json 中导出：
- 每个门/窗 (semantic.type in {"door","window"}) 的 4 个 plane
- 对应的“宿主墙”两块 wall 平面 (host_wall_plane_pair)
- 估算墙厚度 (estimated_wall_thickness)
- 标准化的 wall_id（按 planeID 排序得到的唯一键）

方法要点：
1) 通过 planeLineMatrix 找出门/窗 plane 所有边 line；
2) 找与这些 line 共边的其它 plane，并筛选 type=="wall"；
3) 统计共边次数 + 法向并行/对向检查，选出最相邻的两块墙面；
4) 由两墙面的 (plane.normal, plane.offset) 估算墙厚度；
"""

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _np(a):  # 小工具
    return np.asarray(a, dtype=float)


def normalized(v: np.ndarray) -> np.ndarray:
    v = _np(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def plane_distance_between(
    pid1: int,
    pid2: int,
    plane_normals: Dict[int, np.ndarray],
    plane_offsets: Dict[int, float],
    parallel_tol: float = 0.1,
) -> Optional[float]:
    """
    估算两平行平面之间的距离。
    平面方程：n·x + d = 0（d为offset），n需单位化。
    若法向未平行（或数据缺失），返回 None。
    """
    if pid1 not in plane_normals or pid2 not in plane_normals:
        return None
    n1 = normalized(plane_normals[pid1])
    n2 = normalized(plane_normals[pid2])

    if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
        return None

    # 若 n2 与 n1 反向，则翻转以便对齐
    d1 = plane_offsets.get(pid1, 0.0)
    d2 = plane_offsets.get(pid2, 0.0)
    if np.dot(n1, n2) < 0:
        n2 = -n2
        d2 = -d2

    # 平行性检验
    if abs(np.dot(n1, n2) - 1.0) > parallel_tol:
        return None

    # 距离公式：|d2 - d1|（在 n 已单位化的前提下）
    return float(abs(d2 - d1))


def find_host_wall_pair(
    door_plane_ids: List[int],
    PLM: np.ndarray,
    plane_types: Dict[int, str],
    plane_normals: Dict[int, np.ndarray],
    plane_offsets: Dict[int, float],
    prefer_opposite_normals: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    基于 plane-line 拓扑关系，找到门/窗 plane 周围最可能的两块 wall 平面。
    返回：(wall_plane_a, wall_plane_b)；若失败则返回 None。
    """
    # 预先建立：每个 line 属于哪些 plane
    line_to_planes = [
        set(np.where(PLM[:, j] == 1)[0].tolist()) for j in range(PLM.shape[1])
    ]

    # 统计候选墙面（与门/窗 plane 共边的所有 wall plane）
    candidates = Counter()
    door_plane_set = set(door_plane_ids)
    for pid in door_plane_ids:
        if pid < 0 or pid >= PLM.shape[0]:
            continue
        lines = np.where(PLM[pid] == 1)[0].tolist()
        for lj in lines:
            neighbor = line_to_planes[lj] - door_plane_set
            for npid in neighbor:
                if plane_types.get(int(npid)) == "wall":
                    candidates[int(npid)] += 1

    if not candidates:
        return None

    # 先取共边最多的前若干个 wall plane，再根据法向“对向性”挑一对
    top = [pid for pid, _ in candidates.most_common(6)]
    if len(top) == 1:
        return None
    # 以共边最多的作为锚点，找与其法向相反（dot≈-1）的另一块
    anchor = top[0]
    n0 = normalized(plane_normals.get(anchor, np.zeros(3)))
    best = None
    best_score = (
        -np.inf
    )  # 我们用 -|dot| 作为“相反度”，越接近 -1 越好（即 dot 越接近 -1，-|dot| 越小）
    for pid in top[1:]:
        n = normalized(plane_normals.get(pid, np.zeros(3)))
        if np.linalg.norm(n) == 0 or np.linalg.norm(n0) == 0:
            continue
        dotv = float(np.dot(n0, n))
        # 评分：优先选对向（dot≈-1），也可综合考虑候选出现次数
        score = -abs(dotv) + 1e-4 * candidates[pid]
        if score > best_score:
            best_score = score
            best = pid

    if best is None:
        # 退化策略：就取出现次数最多的前2个
        best = top[1]

    return (int(anchor), int(best))


def export_mapping(
    ann_path: str,
    out_json: str = "door_window_wall_mapping.json",
    out_csv: str = "door_window_wall_mapping.csv",
) -> Dict[str, Any]:
    # 读取注释
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # 必要字段
    for k in ["planes", "semantics", "planeLineMatrix"]:
        if k not in ann:
            raise KeyError(f"annotation_3d.json 缺少必要字段: '{k}'")

    planes = ann["planes"]
    semantics = ann["semantics"]
    PLM = np.array(ann["planeLineMatrix"], dtype=int)

    # 平面类型 / 法向 / 偏移
    plane_types = {int(p["ID"]): p.get("type", None) for p in planes}
    plane_normals = {
        int(p["ID"]): np.array(p.get("normal", [0, 0, 1]), dtype=float) for p in planes
    }
    plane_offsets = {int(p["ID"]): float(p.get("offset", 0.0)) for p in planes}

    rows = []
    for s_idx, sem in enumerate(semantics):
        t = str(sem.get("type", "")).lower()
        if t not in ("door", "window"):
            continue
        pids = [int(x) for x in sem.get("planeID", [])]
        pair = find_host_wall_pair(pids, PLM, plane_types, plane_normals, plane_offsets)
        thickness = None
        wall_id = None
        if pair is not None:
            # 估计墙厚
            thickness = plane_distance_between(
                pair[0], pair[1], plane_normals, plane_offsets
            )
            # 生成稳定的 wall_id（按 planeID 排序的 tuple）
            wall_id = tuple(sorted(pair))

        rows.append(
            {
                "semantic_index": s_idx,
                "semantic_type": t,
                "planeIDs": pids,
                "host_wall_plane_pair": pair,  # e.g. (14, 30)
                "wall_id": wall_id,  # e.g. (14, 30) sorted
                "estimated_wall_thickness": None
                if thickness is None
                else round(float(thickness), 3),
            }
        )

    # 保存 JSON
    result = {
        "apertures": rows,  # 门窗列表
        "meta": {
            "input": ann_path,
            "num_apertures": len(rows),
            "note": "aperture = door or window; wall_id is sorted host_wall_plane_pair",
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存 CSV（扁平化些）
    flat = []
    for r in rows:
        flat.append(
            {
                "semantic_index": r["semantic_index"],
                "semantic_type": r["semantic_type"],
                "planeIDs": ",".join(map(str, r["planeIDs"])) if r["planeIDs"] else "",
                "host_wall_plane_a": (
                    r["host_wall_plane_pair"][0] if r["host_wall_plane_pair"] else ""
                ),
                "host_wall_plane_b": (
                    r["host_wall_plane_pair"][1] if r["host_wall_plane_pair"] else ""
                ),
                "wall_id": ",".join(map(str, r["wall_id"])) if r["wall_id"] else "",
                "estimated_wall_thickness": r["estimated_wall_thickness"],
            }
        )
    pd.DataFrame(flat).to_csv(out_csv, index=False, encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Export door/window -> host wall mapping from annotation_3d.json"
    )
    parser.add_argument("--input", required=True, help="Path to annotation_3d.json")
    parser.add_argument(
        "--out_json", default="door_window_wall_mapping.json", help="Output JSON path"
    )
    parser.add_argument(
        "--out_csv", default="door_window_wall_mapping.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    res = export_mapping(args.input, args.out_json, args.out_csv)
    print(f"Done. apertures: {res['meta']['num_apertures']}")
    print(f"JSON: {args.out_json}")
    print(f"CSV : {args.out_csv}")


if __name__ == "__main__":
    main()
