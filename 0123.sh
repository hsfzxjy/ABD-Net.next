sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 0 --criterion xent  --switch-loss 0 --regularizer so --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 1 --criterion xent  --switch-loss 0 --regularizer so --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-1 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-1__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 2 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 3 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 4 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 5 --criterion xent  --switch-loss 0 --regularizer svdo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 6 --criterion xent  --switch-loss 0 --regularizer svdo --dropout none --data-augment crop --penalty-position before &

sleep 280m

sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 0 --criterion xent  --switch-loss 0 --regularizer so --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 1 --criterion xent  --switch-loss 0 --regularizer so --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-1 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-1__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 2 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 3 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 4 --criterion xent  --switch-loss 0 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-2 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-2__sl_0__fcl_False__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 5 --criterion xent  --switch-loss 0 --regularizer svdo --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_nofc_fd_none_nohead_dan_none_nohead --save-dir 0123_reg_log/densenet121_nofc_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_0__fcl_False__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 6 --criterion xent  --switch-loss 0 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
