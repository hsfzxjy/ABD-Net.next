#!/bin/bash

if [ -z "$GPU" ]; then
    GPU=0,1
fi

export GPU=$GPU
echo Using GPU: $GPU

export LOG_DIR="log/abd_best_cub_200_2011_eof"
echo Logging to: $LOG_DIR

cd `git rev-parse --show-toplevel`

function run_script {
    python train.py -s cub_200_2011 -t cub_200_2011 \
        --flip-eval --eval-freq 1 \
        --label-smooth \
        --criterion htri \
        --lambda-htri 1  \
        --data-augment crop random-erase \
        --margin 0.3 \
        --train-batch-size 64 \
        --height 288 \
        --width 288 \
        --optim adam --lr 0.0003 \
        --stepsize 20 40 \
        --gpu-devices $GPU \
        --max-epoch 120 \
        --save-dir $LOG_DIR \
        --arch resnet50_mgn_like \
        --branches global np2 np3 \
        --global-dim 1024 \
        --np-dim 1024 --np-with-global \
        --use-of \
        --abd-dan cam pam \
        --abd-dim 1024 \
        --cls-dim 2048 \
        --abd-np 1 \
        --use-ow \
        --resnet-last-stride 1 \
        --of-beta 1e-8 
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