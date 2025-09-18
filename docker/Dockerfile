FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
LABEL maintainer="pzhang"

ARG DEBIAN_FRONTEND=noninteractive

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]


ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# If set to nothing, will install the latest version
ARG PYTORCH='2.7.0'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu128'

RUN apt update
RUN apt install -y python3 python3-pip
RUN apt install -y libxext6 libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir -U pip packaging wheel setuptools

RUN pip install --no-cache-dir -U torch==$PYTORCH torchvision --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN pip install --no-cache-dir transformers==4.46.1 safetensors==0.4.5
RUN pip install --no-cache-dir pandas==2.2.3 einops==0.8.1 numpy==1.26.4 \
      scipy==1.15.2 scikit-learn==1.6.1 toml==0.10.2 \
      tokenizers==0.20.3 huggingface-hub==0.34.3 \
      shapely==2.0.7 terminaltables==3.1.10 \
      open3d==0.18.0 laspy ezdxf

RUN pip install --no-cache-dir ninja flash-attn==2.8.3 --no-build-isolation
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
RUN pip install --no-cache-dir timm spconv-cu120

# training
RUN pip install --no-cache-dir omegaconf "datasets<=3.6.0" "accelerate<=1.7.0" wandb

RUN pip uninstall -y tensorflow flax


# Set the working directory
WORKDIR /app


