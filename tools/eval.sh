#!/usr/bin/env bash

test_pcd=data/HC3D/20251027/pcd_test
test_csv=data/HC3D/20251027/test.csv
gt_dir=data/HC3D/20251027/layout
out_dir=data/HC3D/20251027/pred/
#
python inference.py -p $test_pcd -o $out_dir \
    -m work_dirs/hc3d_exp_02/spatiallm-0.5b-sft-hc3d-g1280-e500/checkpoint-8500 \
    -d arch

# Evaluate
python eval.py --metadata test_csv --gt_dir $gt_dir \
    --pred_dir $out_dir  --only_layout
