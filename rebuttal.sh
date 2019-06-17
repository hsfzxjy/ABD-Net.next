#!/bin/bash
python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --gpu-devices 0 --max-epoch 80 --save-dir cam_log/cam --arch resnet50 --branches abd --abd-dan cam --abd-np 1 &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 80 --save-dir cam_log/of_35 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 35 --of-position before &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 80 --save-dir cam_log/of_45 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 45 --of-position before --abd-dan cam &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 2 --max-epoch 80 --save-dir cam_log/of_cam_25 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 25 --of-position before cam --abd-dan cam &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 3 --max-epoch 80 --save-dir cam_log/of_cam_45 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 45 --of-position before cam --abd-dan cam &

wait

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 80 --save-dir cam_log/of_25 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 25 --of-position before &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 80 --save-dir cam_log/of_45 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 45 --of-position before --abd-dan cam &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 2 --max-epoch 80 --save-dir cam_log/of_cam_35 --arch resnet50 --branches abd --abd-np 1 --use-of --of-beta 1e-7 --of-start-epoch 35 --of-position before cam --abd-dan cam &


wait

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 120 --save-dir abd_log/abd --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after cam pam before --abd-dan cam pam &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 120 --save-dir abd_log/abd_swap --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position before --abd-dan cam pam &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 2 --max-epoch 120 --save-dir abd_log/abd_swap_2 --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position before --abd-dan cam pam &

wait

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 120 --save-dir of_type_log/SRIP --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type SRIP &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 120 --save-dir of_type_log/SO --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type SO &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 2 --max-epoch 120 --save-dir of_type_log/DSO --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type DSO &

wait

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 120 --save-dir of_type_log/SRIP_2 --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type SRIP &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 120 --save-dir of_type_log/SO_2 --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type SO &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 2 --max-epoch 120 --save-dir of_type_log/DSO_2 --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after --abd-dan cam pam --of-type DSO &
