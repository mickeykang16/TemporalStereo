
python3 projects/TemporalStereo/demo.py\
 --config-file projects/TemporalStereo/configs/mvsec_single.yaml\
 --checkpoint-path exps/TemporalStereo/mvsec/split1_single/epoch=003.ckpt\
 --resize-to-shape 272 352\
 --data-type MVSEC\
 --data-root /home/jaeyoung/data/ws/event_stereo_ICCV2019/dataset/\
 --annfile /home/jaeyoung/data/ws/event_stereo_ICCV2019/dataset/view_4_test_v5_split1.json\
 --device cuda\
 --log-dir exps/output\
 --frame-index '0'
echo done!