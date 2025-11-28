# 数据准备


## Structure3D

1. 生成点云
2. 生成layout
3. 裁剪数据（移除地面和天花板）


## HC3D

| 序号 | 描述                                 | 对应脚本             |
| ---- | ------------------------------------ | -------------------- |
| 1    | 从数据目录获取信息                   | `fetch_hc3d_data.py` |
| 2    | 将las和dxf数据转换为ply和txt格式数据 | `convert_hc3d.py`    |
| 3    | 生成SpatialLM训练所需的数据源格式    | `gen_train_meta.py`  |
| 4    | 生成评测角点和线段精度的文件         | `gen_eval_config.py` |
