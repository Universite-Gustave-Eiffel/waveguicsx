xhost +local:
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t fenicsx0.6.0 -f DockerFile .
docker run --name fenicsx0.6.0 -ti --rm --ipc=host --network=host --env="DISPLAY" \
    --mount type=bind,source=.,target=/home/waveguicsxuser \
    fenicsx0.6.0
