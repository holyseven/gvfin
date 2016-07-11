LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log

python train.py 2>&1 | tee $LOG
