# One image for both the api and worker services (docker-compose.yml runs
# each with a different command). Not explicitly requested in the file list,
# but docker-compose's `build:` needs an image to build.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY tibetan_layer_detection ./tibetan_layer_detection
COPY api ./api

# CPU by default. For a GPU build pass a CUDA wheel index, e.g.
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
# (docker-compose.gpu.yml does this). The CUDA runtime comes inside the torch
# wheel; the host only needs an NVIDIA driver and the NVIDIA Container Toolkit.
ARG TORCH_INDEX=
RUN if [ -n "$TORCH_INDEX" ]; then pip install --no-cache-dir torch --index-url "$TORCH_INDEX"; fi && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir -r api/requirements.txt

# /data is where book_path (in a POST /process request) is resolved from;
# docker-compose.yml bind-mounts ./data here.
RUN mkdir -p /data
