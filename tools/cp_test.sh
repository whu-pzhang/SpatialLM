#!/user/bin/env bash
set -x 

data_root=data/HC3D/20251027
mkdir -p $data_root/pcd_ff

awk -F',' 'NR>1 {split($2,a,"/");print a[2]}' ${data_root}/test.csv | \
    while read -r line; do
        cp ${data_root}/pcd/${line} $data_root/pcd_test/
    done
