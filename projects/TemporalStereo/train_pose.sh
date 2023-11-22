tensorboard --logdir=exps/TemporalStereo/mvsec/pose4/split3_dilate --port ${TNSBPT} --host=0.0.0.0&
python3 projects/TemporalStereo/dist_train.py  \
    --config-file projects/TemporalStereo/configs/mvsec_4_dilate.yaml
# python3 projects/TemporalStereo/dist_train.py  --config-file projects/TemporalStereo/configs/mvsec_4.yaml
chmod +777 -R exps/TemporalStereo/
echo done!