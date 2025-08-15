# SpatialLM 模型性能测试


## SpatialLM-Testset 数据

```bash
HF_ENDPOINT=https://hf-mirror.com hf download manycore-research/SpatialLM-Testset --repo-type dataset --local-dir data/SpatialLM-Testset
```

评测命令

```bash
python inference.py --point_cloud data/SpatialLM-Testset/pcd --output data/SpatialLM-Testset/pred --model_path manycore-research/SpatialLM1.1-Qwen-0.5B --inference_dtype float32

python eval.py --metadata data/SpatialLM-Testset/test.csv --gt_dir data/SpatialLM-Testset/layout --pred_dir data/SpatialLM-Testset/pred --label_mapping data/SpatialLM-Testset/benchmark_categories.tsv
```

| Layouts | F1 @.25 IoU        | F1 @.50 IoU         |
| ------- | ------------------ | ------------------- |
| wall    | 0.67115267513398   | 0.6118742972069086  |
| door    | 0.5125863385896371 | 0.4398492991950936  |
| window  | 0.4206959706959707 | 0.29911477411477416 |

| Objects         | F1 @.25 IoU         | F1 @.50 IoU          |
| --------------- | ------------------- | -------------------- |
| curtain         | 0.3595630991670596  | 0.10781078107810782  |
| nightstand      | 0.6630952380952382  | 0.30178571428571427  |
| chandelier      | 0.45476190476190476 | 0.23904761904761906  |
| wardrobe        | 0.35714285714285715 | 0.16071428571428573  |
| bed             | 0.9682539682539683  | 0.9523809523809523   |
| sofa            | 0.6542635658914728  | 0.3581395348837209   |
| chair           | 0.24182336182336184 | 0.08222222222222222  |
| cabinet         | 0.15                | 0.06944444444444445  |
| dining table    | 0.28205128205128205 | 0.1282051282051282   |
| plants          | 0.2523809523809524  | 0.16071428571428573  |
| tv cabinet      | 0.42857142857142855 | 0.10714285714285714  |
| coffee table    | 0.48717948717948717 | 0.28205128205128205  |
| side table      | 0.05                | 0.0                  |
| air conditioner | 0.19230769230769232 | 0.038461538461538464 |
| dresser         | 0.4117647058823529  | 0.29411764705882354  |
| stool           | 0.05                | 0.0                  |
| refrigerator    | 0.0                 | 0.0                  |
| painting        | 0.3515248796147673  | 0.08089887640449438  |
| carpet          | 0.26666666666666666 | 0.14411764705882354  |
| tv              | 0.18                | 0.06                 |


## Structured3d-spatiallm 数据

```bash
HF_ENDPOINT=https://hf-mirror.com hf download ysmao/structured3d-spatiallm --repo-type dataset --local-dir data/structured3d-spatiallm
```

生成 metadata

```bash
python data_preprocess/structured3d/generate_metadata.py data/structured3d-spatiallm
```


评测命令

```bash
python inference.py --point_cloud data/structured3d-spatiallm/pcd --output data/structured3d-spatiallm/pred --model_path manycore-research/SpatialLM1.1-Qwen-0.5B --inference_dtype float32 --detect_type arch

python eval.py --metadata data/structured3d-spatiallm/test.csv --gt_dir data/structured3d-spatiallm/layout --pred_dir data/structured3d-spatiallm/pred --label_mapping data/SpatialLM-Testset/benchmark_categories.tsv
```

| Layouts | F1 @.25 IoU        | F1 @.50 IoU         |
| ------- | ------------------ | ------------------- |
| wall    | 0.626064405502135  | 0.5506005462184016  |
| door    | 0.233553664461133  | 0.18595688149501152 |
| window  | 0.4175602793093983 | 0.18967194225466255 |

不同设备上测试结果存在较大差异，但A100上表现更优，和官方结果一致。

| Metrics             | wall | door | window |
| ------------------- | ---- | ---- | ------ |
| F1@.25IoU(5090)     | 62.6 | 23.4 | 41.8   |
| F1@.25IoU(A100)     | 74.3 | 40.1 | 49.1   |
| F1@.25IoU(Official) | 71.5 | 33.3 | 48.7   |
| F1@.50IoU(5090)     | 55.1 | 18.6 | 19.0   |
| F1@.50IoU(A100)     | 68.6 | 34.9 | 20.9   |
| F1@.50IoU(Official) | 64.8 | 29.0 | 21.0   |