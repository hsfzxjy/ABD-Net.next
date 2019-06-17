#!/bin/bash
python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop,random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --gpu-devices 0 --max-epoch 80 --save-dir cam_log/cam --arch resnet50 --branches abd --abd-dan cam --abd-np 1
