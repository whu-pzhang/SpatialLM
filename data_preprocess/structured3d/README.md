## USAGE

1. Convert Structured3d annotations_3d.json to txt
2. Generate train.csv and test.csv from structured3d-spatiallm's "split.csv"
3. Convert layout/*.txt to structured3d_train.json and structured3d_test.json


## Data Origanization

```
scene_<sceneID>
├── 2D_rendering
│   └── <roomID>
│       ├── panorama
│       │   ├── <empty/simple/full>
│       │   │   ├── rgb_<cold/raw/warm>light.png
│       │   │   ├── semantic.png
│       │   │   ├── instance.png
│       │   │   ├── albedo.png
│       │   │   ├── depth.png
│       │   │   └── normal.png
│       │   ├── layout.txt
│       │   └── camera_xyz.txt
│       └── perspective
│           └── <empty/full>
│               └── <positionID>
│                   ├── rgb_rawlight.png
│                   ├── semantic.png
│                   ├── instance.png
│                   ├── albedo.png
│                   ├── depth.png
│                   ├── normal.png
│                   ├── layout.json
│                   └── camera_pose.txt
├── bbox_3d.json
└── annotation_3d.json
```

## Annotation Format

`annotation_3d.json` 结构标注：

```
{
  // PRIMITVIES
  "junctions":[
    {
      "ID":             : int,
      "coordinate"      : List[float]       // 3D vector
    }
  ],
  "lines": [
    {
      "ID":             : int,
      "point"           : List[float],      // 3D vector
      "direction"       : List[float]       // 3D vector
    }
  ],
  "planes": [
    {
      "ID":             : int,
      "type"            : str,              // ceiling, floor, wall
      "normal"          : List[float],      // 3D vector, the normal points to the empty space
      "offset"          : float
    }
  ],
  // RELATIONSHIPS
  "semantics": [
    {
      "ID"              : int,
      "type"            : str,              // room type, door, window
      "planeID"         : List[int]         // indices of the planes
    }
  ],
  "planeLineMatrix"     : Matrix[int],      // matrix W_1 where the ij-th entry is 1 iff l_i is on p_j
  "lineJunctionMatrix"  : Matrix[int],      // matrix W_2 here the mn-th entry is 1 iff x_m is on l_nj
  // OTHERS
  "cuboids": [
    {
      "ID":             : int,
      "planeID"         : List[int]         // indices of the planes
    }
  ]
  "manhattan": [
    {
      "ID":             : int,
      "planeID"         : List[int]         // indices of the planes
    }
  ]
}
```

## 📐 墙体、门窗导出需求规范

### 1. 数据来源
- 输入文件：Structured3D 格式的 `annotation_3d.json`  
- 核心字段：
  - `semantics`：描述房间（如 livingroom、bedroom）的组成平面
  - `planes`：定义平面，带有 `type` 属性（wall / outwall / floor / ceiling / door / window 等）
  - `junctions` + `lines` + `planeLineMatrix`：用于解析平面边界几何

---

### 2. 墙体表示 (Wall)
- **仅包含实体内墙**
  - 从房间 `semantics` 中筛选 `type=="wall"` 的平面  
  - 排除 `outwall`（外墙）、虚拟墙面（开口边界用的辅助面）
- **完整矩形表示**
  - 每个墙体输出为一个完整矩形面（不做门窗减孔）  
  - 底边由该平面在水平面的最小/最大范围与房间地面高度相交确定  
  - 墙高取该房间的实际高度（floor → ceiling）
- **输出格式**
  ```text
  wall_k = Wall(ax, ay, az, bx, by, bz, height, thickness)
  ```
  - `(ax,ay,az)` → 底边起点  
  - `(bx,by,bz)` → 底边终点  
  - `height` → 房间高度  
  - `thickness` → 墙厚（当前默认 0.0，可扩展）

---

### 3. 门和窗表示 (Door / Window)
- **几何来源**
  - 门窗由 `semantics` 中 `type=="door"/"window"` 的四个平面解析  
  - 利用所有角点的 Z 坐标 (`minZ, maxZ`) 得到实际高度与中心位置
- **挂靠墙体**
  - 门窗通过平面邻接关系挂到对应的墙 `wall_k` 上  
  - 输出时需指定 `wall_id`
- **输出格式**
  ```text
  door_i   = Door(wall_id, cx, cy, cz, width, height)
  window_j = Window(wall_id, cx, cy, cz, width, height)
  ```
  - `(cx,cy,cz)` → 开口中心位置  
  - `width` → 左右 jamb 面之间的距离  
  - `height` → `maxZ - minZ`（实际开口高度）

---

### 4. 房间级别处理
- **逐房间计算高度**
  - 每个房间独立计算 `z_floor` 和 `z_ceiling`  
  - 墙体高度以该房间为准，而非全局统一
- **逐房间导出结果**
  - 每个房间的 `Wall/Door/Window` 集合单独输出，便于后续使用

---





## 参考

- [Requesting clarification regarding structured3D result in the paper](https://github.com/manycore-research/SpatialLM/issues/73)
