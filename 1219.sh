#!/bin/bash

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir htri_log/densenet121_fc512_fd_none_nohead_dan_none_nohead_htri_60_10 --gpu-devices 0 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir htri_log/densenet121_fc512_fd_none_nohead_dan_none_nohead_htri_60_0 --gpu-devices 0 --criterion htri &

---

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_htri_60_10 --gpu-devices 0 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_htri_60_10 --gpu-devices 0 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_htri_60_10 --gpu-devices 2 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_htri_60_10 --gpu-devices 2 --criterion htri &

# ---

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_htri_reg_so_60_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_so_60_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_htri_reg_so_60_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_htri_reg_so_60_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer so &

# ---

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_htri_reg_so_60_10 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svmo_60_10 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer svmo &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_htri_reg_svmo_60_10 --gpu-devices 6 --criterion htri --switch-loss 15 --regularizer svmo &

beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_htri_reg_svmo_60_10 --gpu-devices 6 --criterion htri --switch-loss 15 --regularizer svmo &

# ---
sleep 180m

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_htri_reg_svdo_60_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_60_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_htri_reg_svdo_60_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_htri_reg_svdo_60_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo &
