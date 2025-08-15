
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




## 参考

- [Requesting clarification regarding structured3D result in the paper](https://github.com/manycore-research/SpatialLM/issues/73)
