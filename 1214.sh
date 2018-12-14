#!/bin/bash

# 1214

# densenet121 nofc fd cam pam singular

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_c_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_ab_c_head_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_c_head_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_ab_c_head_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_ab_head_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_head_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_ab_head_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_a_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_a_head_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_a_head_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_a_head_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss 15  &



beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_c_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_ab_c_nohead_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_c_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_ab_c_nohead_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_ab_nohead_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_ab_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_ab_nohead_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_a_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_a_nohead_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_a_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_a_head_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss 15  &


sleep 120m

# densenet121 fc512 fd cam pam singlar

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_c_head_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_c_head_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_a_head_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_a_head_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss 15  &



beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_a_nohead_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_a_head_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss 15  &

sleep 120m

# densenet121 fc512 fd cam pam xent

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_c_head_dan_cam_pam_nohead_xent_40_10 --gpu-devices 0 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_xent_40_10 --gpu-devices 0 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_head_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_a_head_dan_cam_pam_nohead_xent_40_10 --gpu-devices 2 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_c_head_dan_cam_pam_head_xent_40_10 --gpu-devices 2 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_xent_40_10 --gpu-devices 3 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_head_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_a_head_dan_cam_pam_head_xent_40_10 --gpu-devices 3 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_nohead_xent_40_10 --gpu-devices 4 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_nohead_xent_40_10 --gpu-devices 4 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_a_nohead_dan_cam_pam_nohead_xent_40_10 --gpu-devices 5 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_c_nohead_dan_cam_pam_head_xent_40_10 --gpu-devices 5 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_ab_nohead_dan_cam_pam_head_xent_40_10 --gpu-devices 6 --criterion xent  &

nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_a_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_a_nohead_dan_cam_pam_head_xent_40_10 --gpu-devices 6 --criterion xent  &

sleep 120m

# densenet121 fc512 cam pam singuar

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_singular_1e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=1e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_singular_1e-9_40_10 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_head_singular_5e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

beta=5e-9 nohup python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead_singular_5e-9_40_10 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15  &

# densenet121 fc512 pam cam reg

beta=1e-4 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_xent_reg_so_1e-4_40_10 --gpu-devices 3 --criterion xent --regularizer so --switch-loss 15 &

beta=1e-4 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_xent_reg_so_1e-4_40_10 --gpu-devices 3 --criterion xent --regularizer so --switch-loss 15 &

beta=1e-4 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_xent_reg_svmo_1e-4_40_10 --gpu-devices 4 --criterion xent --regularizer svmo --switch-loss 15 &

beta=1e-4 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_xent_reg_svmo_1e-4_40_10 --gpu-devices 4 --criterion xent --regularizer svmo --switch-loss 15 &

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_xent_reg_svdo_1e-6_40_10 --gpu-devices 5 --criterion xent --regularizer svdo --switch-loss 15 &

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_xent_reg_svdo_1e-6_40_10 --gpu-devices 5 --criterion xent --regularizer svdo --switch-loss 15 &

# densenet121 nofc cam pam reg

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_head_xent_reg_so_1e-6_40_10 --gpu-devices 6 --criterion xent --regularizer so --switch-loss 15 &

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead_xent_reg_so_1e-6_40_10 --gpu-devices 6 --criterion xent --regularizer so --switch-loss 15 &

beta=1e-3 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_head_xent_reg_svmo_1e-3_40_10 --gpu-devices 7 --criterion xent --regularizer svmo --switch-loss 15 &

beta=1e-3 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead_xent_reg_svmo_1e-3_40_10 --gpu-devices 7 --criterion xent --regularizer svmo --switch-loss 15 &

sleep 120m

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_head --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_head_xent_reg_svdo_1e-6_40_10 --gpu-devices 5 --criterion xent --regularizer svdo --switch-loss 15 &

beta=1e-6 nohup python train_reg.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir 1214_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead_xent_reg_svdo_1e-6_40_10 --gpu-devices 5 --criterion xent --regularizer svdo --switch-loss 15 &
