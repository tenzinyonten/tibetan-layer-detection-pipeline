# One image for both the api and worker services (docker-compose.yml runs
# each with a different command). Not explicitly requested in the file list,
# but docker-compose's `build:` needs an image to build.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY tibetan_layer_detection ./tibetan_layer_detection
COPY api ./api

RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir -r api/requirements.txt

# /data is where book_path (in a POST /process request) is resolved from;
# docker-compose.yml bind-mounts ./data here.
RUN mkdir -p /data
