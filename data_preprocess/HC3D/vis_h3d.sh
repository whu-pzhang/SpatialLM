#!/usr/bin/env bash

set -x

# argv0: input point cloud directory
# argv1: input layout directory
# argv2: output directory

pcd_dir=${1}
layout_dir=${2}
output_dir=${3}

mkdir -p ${output_dir}

for pcd_file in ${pcd_dir}/*.ply; do
    scene_name=$(basename ${pcd_file} .ply)
    layout_file=${layout_dir}/${scene_name}.txt
    output_file=${output_dir}/${scene_name}.rrd
    python visualize.py --point_cloud ${pcd_file} --layout ${layout_file} --save ${output_file} --radius 0.005 --hide-labels
done
