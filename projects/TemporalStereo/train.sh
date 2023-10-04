# docker build -t temporal_stereo:test -f /home/jaeyoung/docker/Dockerfile . 
# docker run -it --gpus '"device=4,5,6,7"' temporal_stereo:test bash


python3 -m debugpy --listen 0.0.0.0:5678 /home/jaeyoung/ws/TemporalStereo/projects/TemporalStereo/dist_train.py --config-file /home/jaeyoung/ws/TemporalStereo/projects/TemporalStereo/configs/kitti2015-pair.yaml

echo Results are saved to $LOGDIR.
echo done!