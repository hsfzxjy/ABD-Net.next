#!/bin/bash
python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 0 --gpu-devices 0 --max-epoch 120 --save-dir abd_log/abd --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-start-epoch 0 --of-type SRIP
