#!/usr/bin/env bash

txt_dir=data/HC3D/20251118/val_pred_pretrain_2_5cm
dxf_dir=data/HC3D/20251118/val_pred_pretrain_2_5cm/dxf

mkdir -p $dxf_dir

for txt_file in $txt_dir/*.txt; do
    base_name=$(basename "$txt_file" .txt)
    dxf_file="$dxf_dir/${base_name}.dxf"
    python tools/txt2dxf_2d.py "$txt_file" -o "$dxf_file" -v
done
echo "Conversion from TXT to DXF completed."
