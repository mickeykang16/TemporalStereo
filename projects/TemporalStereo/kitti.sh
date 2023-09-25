CONFIG=./configs/kitti2015-multi.yaml
CKPT=./exps/kitti2015_multi.ckpt
LOGDIR=./exps/output/
DATA_ROOT=/home/jaeyoung/data/
DATA_TYPE=KITTI2015
ANNFILE='/home/jaeyoung/data/KITTI-2015/view_2_train_12&15.json'
# H=544
# W=960
H=375
W=1242
DEVICE=cuda

echo Starting running demo...

CUDA_VISIBLE_DEVICES=4,5,6,7 python demo.py  --config-file $CONFIG \
                --checkpoint-path $CKPT \
                --resize-to-shape $H $W \
                --data-type $DATA_TYPE \
                --data-root  $DATA_ROOT\
                --annfile $ANNFILE \
                --device $DEVICE \
                --log-dir $LOGDIR

echo Results are saved to $LOGDIR.
echo done!