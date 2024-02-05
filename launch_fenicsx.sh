#!/bin/bash 

# ================
# This bash scripts builds a docker image based on dolfinx/dolfinx:v0.6.0
# ================

xhost +local:
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t dolfinx0.6.0 -f DockerFile .
docker run --name dolfinx0.6.0 -ti --rm --ipc=host --network=host --env="DISPLAY" \
    --mount type=bind,source=.,target=/home/waveguicsxuser \
    dolfinx0.6.0
