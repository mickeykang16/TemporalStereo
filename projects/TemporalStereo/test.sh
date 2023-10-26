
python3 -m pdb -c continue /home/jaeyoung/ws/TemporalEventStereo/projects/TemporalStereo/demo.py\
 --config-file /home/jaeyoung/ws/TemporalEventStereo/projects/TemporalStereo/configs/kitti2015-pair.yaml\
 --checkpoint-path /home/jaeyoung/ws/TemporalEventStereo/projects/TemporalStereo/exps/sceneflow.ckpt\
 --resize-to-shape 384 1280\
 --data-type KITTI2015\
 --data-root /home/jaeyoung/data/\
 --annfile '/home/jaeyoung/data/KITTI-2015/view_2_train_12&15.json'\
 --device cuda\
 --log-dir /home/jaeyoung/ws/TemporalEventStereo/projects/TemporalStereo/exps/output\
 --frame-index '-1' '0'
echo done!