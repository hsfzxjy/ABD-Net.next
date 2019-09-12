#!/bin/bash

if [ -z "$GPU" ]; then
    GPU=0
fi

if [ -z "$of_beta" ]; then
    of_beta=1e-6
fi

export of_beta=$of_beta
echo Using OF Beta: $of_beta

export GPU=$GPU
echo Using GPU: $GPU

export LOG_DIR="log/resnet_of_${of_beta}_market_eof"
echo Logging to: $LOG_DIR

cd `git rev-parse --show-toplevel`

function run_script {
    python train.py -s market1501 -t market1501 \
        --flip-eval --eval-freq 1 \
        --label-smooth \
        --criterion xent \
        --data-augment crop \
        --margin 0.3 \
        --train-batch-size 32 \
        --height 384 \
        --width 128 \
        --optim adam --lr 0.0003 \
        --stepsize 20 40 \
        --gpu-devices $GPU \
        --max-epoch 60 \
        --save-dir $LOG_DIR \
        --arch resnet50 \
        --branches abd \
        --use-of \
        --of-position before \
        --abd-dim 1024 \
        --abd-np 1 \
        --resnet-last-stride 1 \
        --of-beta $of_beta --of-start-epoch 25
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