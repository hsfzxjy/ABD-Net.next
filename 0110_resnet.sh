nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_none_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_none_head_htri_60_10_size_256__0 --gpu-devices 0 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_none_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_none_head_xent_60_10_size_256__0 --gpu-devices 1 --criterion xent &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_none_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_none_head_htri_60_10_size_256__0 --gpu-devices 2 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_none_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_none_head_xent_60_10_size_256__0 --gpu-devices 3 --criterion xent &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_60_10_size_256__0 --gpu-devices 4 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_60_10_size_256__0 --gpu-devices 5 --criterion xent &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_60_10_size_256__0 --gpu-devices 6 --criterion htri &

nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_60_10_size_256__0 --gpu-devices 7 --criterion xent &

sleep 180m

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_256__0 --gpu-devices 0 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_256_position_cam_0 --gpu-devices 1 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_256_position_cam_0 --gpu-devices 2 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_256_position__0 --gpu-devices 3 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_256__0 --gpu-devices 5 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_256_position_cam_0 --gpu-devices 6 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_256_position_cam_0 --gpu-devices 4 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_256_position__0 --gpu-devices 7 --criterion singular --switch-loss 15 --fix-custom-loss &

sleep 300m

beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-5_60_10_size_256__0 --gpu-devices 0 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-5_60_10_size_256_position_cam_0 --gpu-devices 1 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-5_60_10_size_256_position_cam_0 --gpu-devices 2 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-5_60_10_size_256_position__0 --gpu-devices 3 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-8 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-8_60_10_size_256__0 --gpu-devices 5 --criterion singular --switch-loss 15 --fix-custom-loss &

beta=1e-8 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-8_60_10_size_256_position_cam_0 --gpu-devices 6 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-8 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-8_60_10_size_256_position_cam_0 --gpu-devices 4 --criterion singular --switch-loss 15 --fix-custom-loss --penalty-position cam &

beta=1e-8 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-8_60_10_size_256_position__0 --gpu-devices 7 --criterion singular --switch-loss 15 --fix-custom-loss &

sleep 300m

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_256_market__0 --gpu-devices 0 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_256_market__0 --gpu-devices 1 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_256_market__0 --gpu-devices 2 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_256_market__0 --gpu-devices 3 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_256_market__0 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_256_market__0 --gpu-devices 5 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_256_market__0 --gpu-devices 6 --criterion htri --switch-loss 15 --regularizer svdo &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_256_market__0 --gpu-devices 7 --criterion htri --switch-loss 15 --regularizer svdo &

sleep 180m


beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 0 --criterion htri --switch-loss 15 --regularizer svdo --data-augment crop,random-erase  --dropout incr &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 15 --data-augment crop,random-erase  --dropout incr &

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 2 --criterion singular --fix-custom-loss --switch-loss 15 --data-augment crop,random-erase  --dropout incr &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_ab_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_ab_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 0 --criterion htri --switch-loss 15 --regularizer svdo --data-augment crop,random-erase  --dropout incr &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-6_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 4 --criterion htri --switch-loss 15 --regularizer svdo --data-augment crop,random-erase  --dropout incr &

beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-6_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 15 --data-augment crop,random-erase  --dropout incr &

beta=1e-9 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_xent_singular_1e-9_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss 15 --data-augment crop,random-erase  --dropout incr &

beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 dukemtmcreid -j 4 --height 384 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_fc512_fd_none_nohead_dan_cam_pam_head --save-dir 0110_resnet_log/resnet50_fc512_fd_none_nohead_dan_cam_pam_head_htri_reg_svdo_1e-3_60_10_size_384_dropout_incr_dau_crop,random-erase_market__0 --gpu-devices 7 --criterion htri --switch-loss 15 --regularizer svdo --data-augment crop,random-erase  --dropout incr &