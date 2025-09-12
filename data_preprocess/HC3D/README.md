# HC3D 数据准备

HC3D 为华测自采并标注的室内手持点云数据集，主要是为了室内布局估计任务。

## 原始数据说明

原始数据位于 `\\10.12.11.115\shared\01_3D-FAVP\handheld_scanner_Data\RS10\` 路径下，标注了三维矢量的数据有 10 套：

```
├─Annotations
├─biguiyuanxingzuan_res_ff_RS10_0
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  ├─LAS
│  └─LAS_Refined
├─biguiyuanxingzuan_res_ff_RS10_1
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  ├─LAS
│  ├─LAS_cut
│  ├─LAS_Refined
│  └─LAS_Refined_cut
├─biguiyuanxingzuan_res_ff_RS10_2
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  ├─LAS
│  └─LAS_cut
├─biguiyuanxingzuan_res_ff_RS10_3
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  └─LAS
├─biguiyuanxingzuan_res_ff_RS10_4
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  └─LAS
├─changyinguanlinfu_res_uf_RS10_0
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  ├─LAS
│  └─LAS_cut
├─changyinguanlinfu_res_uf_RS10_1
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  ├─LAS
│  └─LAS_cut
├─changyinguanlinfu_res_uf_RS10_2
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  └─LAS
├─changyinguanlinfu_res_uf_RS10_3
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  └─LAS
├─changyinguanlinfu_res_uf_RS10_4
│  ├─Annotations
│  ├─Camera1
│  ├─Camera2
│  ├─Camera3
│  └─LAS
```

其中，`LAS` 文件夹为原始点云数据，`LAS_Refined` 文件夹为处理后的点云数据，`Annotations` 文件夹为标注文件，其结构如下

```
├─Annotations
│  ├─3D.dxf         # 3D 标注
│  ├─floorplan.dxf  # 2D 平面图标注
```




## 数据处理

1. 检查修改标注
2. 原始 dxf 格式转换为 SpatialLM 所需的 txt 格式
3. 原始 las 点云转换为 ply 格式点云
4. 生成 SpatialLM 训练数据, `HC3D_train.json`
5. 生成数据集元数据，如 `dataset_info.json`


```
data/HC3D/processed
├── dataset_info.json
├── HC3D_test.json
├── HC3D_train.json
├── layout
│   ├── scene_00000.txt
│   ├── scene_00001.txt
│   ├── scene_00002.txt
│   ├── scene_00005.txt
│   ├── scene_00006.txt
│   ├── scene_00007.txt
│   ├── scene_00008.txt
│   └── scene_00009.txt
├── pcd
│   ├── scene_00000.ply
│   ├── scene_00001.ply
│   ├── scene_00002.ply
│   ├── scene_00005.ply
│   ├── scene_00006.ply
│   ├── scene_00007.ply
│   ├── scene_00008.ply
│   └── scene_00009.ply
├── test.csv
└── train.csv
```