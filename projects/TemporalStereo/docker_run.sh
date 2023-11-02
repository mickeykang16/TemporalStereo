# docker run -it --gpus '"device=0,1,2,3"' \
gpu=$1
pn=$((6006+$gpu))
pn2=$((6010+$gpu))
# pn=6010

docker run -it --gpus device=${gpu} \
    --name temporalstereo_${gpu}\
    -p ${pn}:${pn} \
    --shm-size=128G \
    -e TNSBPT=${pn}\
    -e GPUNUM=$gpu\
    -v /home/user/jaeyoung/data:/home/jaeyoung/data\
    -v /home/user/jaeyoung/ws:/home/jaeyoung/ws\
    temporal_stereo:event bash 
    # tmux new-session -s gpu$gpu 'tmux source-file ./.tmux.conf; $SHELL'
    

