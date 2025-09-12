#!/usr/bin/env bash
# GT
# txt_dir="data/HC3D/processed/layout"
# dst_dir="data/HC3D/processed/gt_layout_2d_dxf"
# layer_name="room"
# pred
txt_dir="data/HC3D/processed/pred_test"
dst_dir="data/HC3D/processed/pred_layout_2d_dxf"
layer_name="wall_poly"
# point cloud directory
test_pcd_dir="data/HC3D/processed/pcd"

# Get test scenes
test_scenes=$(ls $test_pcd_dir | sed 's/.ply/.txt/g')

mkdir -p $dst_dir
for txt_file in $txt_dir/*.txt; do
    scene_name=$(basename $txt_file .txt)
    output_file="$dst_dir/$scene_name.dxf"
    if [[ $test_scenes == *"$scene_name"* ]]; then
        echo "Processing $scene_name"
        python tools/txt2dxf.py $txt_file --output $output_file --layer ${layer_name}
    fi
done