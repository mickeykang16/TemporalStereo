tensorboard --logdir=exps/TemporalStereo/mvsec/single_36.4 --port ${TNSBPT} --host=0.0.0.0&
python3 projects/TemporalStereo/dist_train.py  \
    --config-file projects/TemporalStereo/configs/mvsec_single.yaml
# python3 projects/TemporalStereo/dist_train.py  --config-file projects/TemporalStereo/configs/mvsec_4.yaml
chmod +777 exps/TemporalStereo/
echo done!