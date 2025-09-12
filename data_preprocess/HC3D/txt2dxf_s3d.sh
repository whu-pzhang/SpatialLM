#!/usr/bin/env bash
# GT
# txt_dir="data/structured3d-spatiallm-pzhang/layout"
txt_dir="data/structured3d-spatiallm/layout"
dst_dir="data/structured3d-spatiallm-pzhang/layout_test_dxf"
# pred
# txt_dir="data/structured3d-spatiallm-pzhang/pred_test_csft_e50"
# dst_dir="data/structured3d-spatiallm-pzhang/pred_test_csft_e50/dxf"
# #
test_dir="data/structured3d-spatiallm-pzhang/pcd_test"

# Get test scenes
test_scenes=$(ls $test_dir | sed 's/.ply/.txt/g')

mkdir -p $dst_dir
for txt_file in $txt_dir/*.txt; do
    scene_name=$(basename $txt_file .txt)
    output_file="$dst_dir/$scene_name.dxf"
    if [[ $test_scenes == *"$scene_name"* ]]; then
        echo "Processing $scene_name"
        python tools/txt2dxf.py $txt_file --output $output_file --layer room
    fi
done