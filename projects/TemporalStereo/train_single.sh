tensorboard --logdir=exps/TemporalStereo/mvsec/single_compare+ --port ${TNSBPT} --host=0.0.0.0&
python3 -m pdb -c continue projects/TemporalStereo/dist_train.py  \
    --config-file projects/TemporalStereo/configs/mvsec_single.yaml
# python3 projects/TemporalStereo/dist_train.py  --config-file projects/TemporalStereo/configs/mvsec_4.yaml

echo done!