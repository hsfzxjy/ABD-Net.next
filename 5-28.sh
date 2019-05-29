nohup python train_aicity19.py -s cub_200_2011 -t cub_200_2011 --label-smooth --criterion htri --lambda-htri 0.1  --data-augment crop,random-erase --margin 1.2 --train-batch-size 64  --height 224 --width 224 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 15 40  --max-epoch 100 --save-dir cub_log/ABD_1_1_p2_15 --arch resnet50 --gpu-devices 2,0 --use-of --use-ow --branches global abd --abd-dan cam pam --abd-np 2 --eval-freq 1 --print-freq 1 &

wait

nohup python train_aicity19.py -s cub_200_2011 -t cub_200_2011 --label-smooth --criterion htri --lambda-htri 0.1  --data-augment crop,random-erase --margin 1.2 --train-batch-size 64  --height 224 --width 224 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 100 --save-dir cub_log/ABD_1_1_p2_20 --arch resnet50 --gpu-devices 2,0 --use-of --use-ow --branches global abd --abd-dan cam pam --abd-np 2 --eval-freq 1 --print-freq 1 &

wait

nohup python train_aicity19.py -s cub_200_2011 -t cub_200_2011 --label-smooth --criterion htri --lambda-htri 0.1  --data-augment crop,random-erase --margin 1.2 --train-batch-size 64  --height 224 --width 224 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 15 40  --max-epoch 100 --save-dir cub_log/ABD_1_1_p2_15_fb_1e-7 --arch resnet50 --gpu-devices 2,0 --use-of --use-ow --branches global abd --abd-dan cam pam --abd-np 2 --eval-freq 1 --print-freq 1 --of-beta 1e-7 &

wait

nohup python train_aicity19.py -s cub_200_2011 -t cub_200_2011 --label-smooth --criterion htri --lambda-htri 0.1  --data-augment crop,random-erase --margin 1.2 --train-batch-size 64  --height 224 --width 224 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40  --max-epoch 100 --save-dir cub_log/ABD_1_1_p2_20_fb_1e-7 --arch resnet50 --gpu-devices 2,0 --use-of --use-ow --branches global abd --abd-dan cam pam --abd-np 2 --eval-freq 1 --print-freq 1 --of-beta 1e-7 &
