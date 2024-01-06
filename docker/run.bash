# buiding:
# docker build -f docker/dockerfile -t autobrains .

docker run -it \
    -v $PWD/autobrains:/workspace/DeepLense/autobrains \
    -v $PWD/config:/workspace/DeepLense/config \
    -v $PWD/data:/workspace/DeepLense/data \
    -v $PWD/logger:/workspace/DeepLense/logger \
    -v $PWD/scripts:/workspace/DeepLense/scripts \
    -v $PWD/tests:/workspace/DeepLense/tests \
    --gpus all \
    --privileged \
    deeplense:latest bash 
