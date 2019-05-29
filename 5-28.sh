nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --of-beta 1e-7 &

wait

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix_of_0 --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 0 --of-position global --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix_of_0 --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 0 --of-position global --of-beta 1e-7 &

wait

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix_of_10 --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 10 --of-position global --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix_of_10 --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 10 --of-position global --of-beta 1e-7 &

wait

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix_of_20 --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 20 --of-position global --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix_of_20 --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 20 --of-position global --of-beta 1e-7 &

wait

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix_of_30 --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 30 --of-position global --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix_of_30 --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 30 --of-position global --of-beta 1e-7 &

wait

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 0 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/nofix_of_40 --arch resnet50 --gpu-devices 1 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 40 --of-position global --of-beta 1e-7 &

nohup python train.py -s market1501 -t market1501 --label-smooth --criterion xent --lambda-htri 0.1  --data-augment none --margin 1.2 --train-batch-size 32  --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 70 --save-dir of_study/fix_of_40 --arch resnet50 --gpu-devices 2 --branches global --eval-freq 1 --global-dim 0 --use-of --of-start-epoch 40 --of-position global --of-beta 1e-7 &

