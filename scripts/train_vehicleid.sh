#!/bin/bash

if [ -z "$GPU" ]; then
    GPU=0,1
fi

export GPU=$GPU
echo Using GPU: $GPU

export LOG_DIR="log/abd_best_vehicleid_"$2"_eof"

cd `git rev-parse --show-toplevel`

function run_script {
    echo Logging to: $LOG_DIR
    python $script_name -s $dataset_name -t $dataset_name \
        --flip-eval --eval-freq -1 \
        --label-smooth \
        --criterion htri \
        --lambda-htri 0.1  \
        --data-augment crop random-erase \
        --margin 1.2 \
        --train-batch-size 64 \
        --height 224 \
        --width 224 \
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
        --use-ow $extra_args
}
export -f run_script

export script_name=train.py
export dataset_name=vehicleid
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
elif [ "$1"x == "test_multi"x ]; then
    export extra_args="--evaluate --load-weights ${LOG_DIR}/checkpoint_latest.pth.tar"
    old_log_dir=$LOG_DIR
    export LOG_DIR=../.__
    export script_name=tester_multi.py
    for i in 800 1600 2400 3200 6000 13164; do
        export dataset_name=vehicleid_$i
        run_script | tee -a ${old_log_dir}/log_multi_test.txt
    done
else
    echo Unknown sub-command: $1
    exit 1
fi