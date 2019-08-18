#!/bin/bash

if [ -z "$GPU" ]; then
    GPU=0,1
fi

export GPU=$GPU
echo Using GPU: $GPU

export LOG_DIR="log/abd_best_duke_"
echo Logging to: $LOG_DIR

cd `git rev-parse --show-toplevel`

function run_script {
    python train.py -s dukemtmcreid -t dukemtmcreid \
        --flip-eval --eval-freq 1 \
        --label-smooth \
        --criterion htri \
        --lambda-htri 0.1  \
        --data-augment crop random-erase \
        --margin 1.2 \
        --train-batch-size 64 \
        --height 384 \
        --width 128 \
        --optim adam --lr 0.0003 \
        --stepsize 20 40 \
        --gpu-devices $GPU \
        --max-epoch 120 \
        --save-dir $LOG_DIR \
        --arch resnet50 \
        --use-of \
        --abd-dan cam pam \
        --abd-np 2 \
        --shallow-cam \
        --use-ow 
}
export -f run_script

if [ "$1"x == "train"x ]; then
    nohup bash -c run_script > /dev/null &
    sleep 2
    tail -f $LOG_DIR/log_train.txt
elif [ "$1"x == "debug"x ]; then
    run_script
elif [ "$1"x == "test"x ]; then
    export LOG_DIR=../.__
    export extra_args="--evaluate --load-weights $2"
    run_script
elif [ "$1"x == "kill"x ]; then
    ps aux | grep $LOG_DIR | awk '{system("kill " $2)}'
elif [ "$1"x == "log"x ]; then
    tail -f $LOG_DIR/log_train.txt -n 1000
else
    echo Unknown sub-command: $1
    exit 1
fi