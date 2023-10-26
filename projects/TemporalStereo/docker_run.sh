# docker run -it --gpus '"device=0,1,2,3"' \
docker run -it --gpus all \
    --name temporalstereo\
    -p 6006:6006 -p 5678:5678 --shm-size=999G \
    -v /home/user/jaeyoung/data:/home/jaeyoung/data\
    -v /home/user/jaeyoung/ws:/home/jaeyoung/ws\
    temporal_stereo:event bash