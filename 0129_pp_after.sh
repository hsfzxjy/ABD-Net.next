sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before &

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_pam__size_256__0 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position pam &

sleep 150m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_pam__size_256__1 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position pam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_cam__size_256__0 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_cam__size_256__1 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position cam &

sleep 150m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_pam,cam__size_256__0 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position pam,cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_pam,cam__size_256__1 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position pam,cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,pam__size_256__0 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,pam &

sleep 150m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,pam__size_256__1 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,pam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,cam__size_256__0 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,cam__size_256__1 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,cam &

sleep 150m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,pam,cam__size_256__0 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,pam,cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_before,pam,cam__size_256__1 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position before,pam,cam &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_layer5__size_256__0 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position layer5 &

sleep 150m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead --save-dir pp_log/densenet121_nofc_fd_none_nohead_dan_cam_pam_nohead__crit_singular__sb_1e-6__b___sl_12__fcl_True__reg_none__dropout_none__dau_crop__pp_layer5__size_256__1 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss 12 --regularizer none --dropout none --data-augment crop --penalty-position layer5 &



