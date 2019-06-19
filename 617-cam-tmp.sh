python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment none --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 120 --save-dir abd_log/of_cam_none_xent --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after cam before --abd-dan cam  &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion xent --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 120 --save-dir abd_log/of_cam_re_xent --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after cam before --abd-dan cam  &

wait

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment none --margin 2.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 0 --max-epoch 120 --save-dir abd_log/of_cam_none_htri --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after cam before --abd-dan cam  &

python train.py -s market1501 -t market1501 --flip-eval --eval-freq 1 --label-smooth --criterion htri --lambda-htri 0.1 --data-augment crop random-erase --margin 1.2 --train-batch-size 64 --height 384 --width 128 --optim adam --lr 0.0003 --stepsize 20 40 --fixbase-epoch 10 --gpu-devices 1 --max-epoch 120 --save-dir abd_log/of_cam_re_htri --arch resnet50 --branches global abd --abd-np 2 --use-of --of-beta 1e-6 --of-position after cam before --abd-dan cam  &

