
## Sturctured3D 结果复现


官方论文结果如下：

![alt text](./asset/spatiallm_layout.png)

所有指标均在 structured3d 测试数据集上测试，记录 F1@0.5IoU 指标：

| Model                  | FT Datset      | Test Dataset | wall  | door  | window | Avg   | Note     | Config                                                                        |
| ---------------------- | -------------- | ------------ | ----- | ----- | ------ | ----- | -------- | ----------------------------------------------------------------------------- |
| SpatialLM1.1-Qwen-0.5B | spatiallm-data | str3d        | 67.96 | 34.15 | 22.03  | 41.38 | Official |                                                                               |
| SpatialLM1.1-Qwen-0.5B | spatiallm-data | str3d-pzhang | 67.09 | 34.00 | 34.82  | 45.30 | Official |                                                                               |
| SpatialLM1.1-Qwen-0.5B | str3d          | str3d        | 93.46 | 81.52 | 75.69  | 83.56 | Official | [model](https://huggingface.co/ysmao/SpatialLM1.1-Qwen-0.5B-Structured3D-SFT) |
| SpatialLM1.1-Qwen-0.5B | str3d          | str3d-pzhang | 92.16 | 59.29 | 18.96  | 56.80 | Official |                                                                               |
| SpatialLM1.1-0.5B-sft  | str3d-pzhang   | str3d        | 87.53 | 61.48 | 22.96  | 57.32 |          | [config](configs/spatiallm_sft_structured3d_v1.yaml)                          |
| SpatialLM1.1-0.5B-sft  | str3d-pzhang   | str3d-pzhang | 86.56 | 62.23 | 45.21  | 64.67 |          |                                                                               |


Note:
- `SpatialLM1.1-Qwen-0.5B`: SpatialLM 官方模型
- `SpatialLM1.1-0.5B-sft`: 在 `SpatialLM1.1-Qwen-0.5B` 模型基础上，采用 `spatial-structured3d-pzhang` 数据集进行微调的模型
- `spatiallm-data`: 为 SpatialLM 论文中所用到的数据，未公开
- `str3d`: 为作者提供的转换好的 Structured3d 数据集，其中的门和窗高度为所在墙面的高度
- `str3d-pzhang`: 我们重新处理后的 Structured3d 数据集，门和窗的高度为实际高度，同时修复了一些数据错误
- 作者对str3d复现的提示：https://github.com/manycore-research/SpatialLM/issues/79#issuecomment-3146998014

