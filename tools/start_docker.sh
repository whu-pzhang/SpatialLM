#!/usr/bin/env bash

docker run --privileged --gpus all --shm-size 10g --rm -it \
    --name spatiallm --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,src="${HOME}/.cache/huggingface",target=/root/.cache/huggingface \
    -v /mnt/data1/pzhang:/app/data \
    -v /mnt/data1/pzhang/HC3d_work_dirs:/app/work_dirs \
    -v $(pwd):/app \
    -w /app \
    spatiallm:v1.1
