#!/usr/bin/env sh

python -u MLP.py --evaluate \
--train-root '.' \
--val-root '.' \
--batch-size 128 \
--test-batch-size 100 \
--epochs 500 \
--lr 0.01 \
--momentum 0.9 \
--decay_epoch 140 \
--decay_rate 0.2 \
--multi_gpu \
--no-cuda \
--save-path 'checkpointtest/' \
--load-path 'checkpoint/_204.pth.tar' \
 > log_test.txt
#--resume \
#--resume \
#--tsne \