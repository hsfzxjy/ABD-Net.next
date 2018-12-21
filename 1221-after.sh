

#---
 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_htri_80_10 --gpu-devices 3 --criterion htri --switch-loss 15 &

 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_htri_80_10 --gpu-devices 4 --criterion htri --switch-loss 15 &

 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_head_htri_80_10 --gpu-devices 5 --criterion htri --switch-loss 15 &

 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_nohead_htri_80_10 --gpu-devices 6 --criterion htri --switch-loss 15 &

# ---

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_htri_reg_so_80_10 --gpu-devices 7 --criterion htri --switch-loss 15 --regularizer so &

sleep 240m




beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_nohead_htri_reg_so_80_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer so &

# ---

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_htri_reg_so_80_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svmo_80_10 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer svmo &

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_head_htri_reg_svmo_80_10 --gpu-devices 6 --criterion htri --switch-loss 15 --regularizer svmo &

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_nohead_htri_reg_svmo_80_10 --gpu-devices 7 --criterion htri --switch-loss 15 --regularizer svmo &

sleep 240m



beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_head_htri_reg_svdo_80_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_nohead_htri_reg_svdo_80_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_head_htri_reg_svdo_80_10 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_ab_head_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_ab_head_dan_cam_pam_nohead_htri_reg_svdo_80_10 --gpu-devices 6 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_so_80_10 --gpu-devices 7 --criterion htri --switch-loss 15 --regularizer so &

sleep 240m

beta=1e-4  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_head_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_head_dan_cam_pam_head_htri_reg_so_80_10 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer so &

beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_nohead_htri_reg_svdo_80_10 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6  nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 1221_aft_htri_log/densenet121_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_80_10 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer svdo &
