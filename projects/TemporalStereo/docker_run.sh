docker run -it --rm --gpus '"device=4,5,6,7"' \
    -p 6006:6006 -p 5678:5678 --shm-size=999G \
    -v /home/jaeyoung/data/KITTI-2015:/home/jaeyoung/data/KITTI-2015\
    -v /home/jaeyoung/data/ws/TemporalStereo:/home/jaeyoung/ws/TemporalStereo \
    temporal_stereo:test bash