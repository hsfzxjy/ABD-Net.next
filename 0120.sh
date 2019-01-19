sleep 300m

sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-6__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-6__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-6__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__2 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-6 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-6__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__3 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &

sing_beta=1e-7 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-7__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 7 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-7 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-7__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-7 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-7__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__2 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-7 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-7__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__3 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &

sing_beta=1e-8 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-8__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-8 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-8__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 5 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-8 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-8__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__2 --gpu-devices 6 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-8 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-8__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__3 --gpu-devices 7 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &

sleep 210m

sing_beta=1e-9 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-9__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 0 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-9 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-9__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 1 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-9 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-9__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__2 --gpu-devices 3 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &
sing_beta=1e-9 beta= nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_singular__sb_1e-9__b___sl_-15__fcl_True__reg_none__dropout_none__dau_crop__pp_before__size_256__3 --gpu-devices 4 --criterion singular --fix-custom-loss --switch-loss -15 --regularizer none --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 5 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 6 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &

sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 7 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 0 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 1 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 3 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 4 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_so__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 5 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer so --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 6 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 7 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &

sleep 210m

sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 0 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 1 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 3 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 4 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 5 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_svmo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 6 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svmo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 7 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-3__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 0 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 1 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-4 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-4__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 3 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 4 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-5 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-5__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 5 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__0 --gpu-devices 6 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
sing_beta= beta=1e-6 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512_fd_none_nohead_dan_none_nohead --save-dir log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b_1e-6__sl_-15__fcl_True__reg_svdo__dropout_none__dau_crop__pp_before__size_256__1 --gpu-devices 7 --criterion xent --fix-custom-loss --switch-loss -15 --regularizer svdo --dropout none --data-augment crop --penalty-position before &
