sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-35__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_60__ucg_False__cg_0.5__0 --gpu-devices 0 --criterion singular_htri  --switch-loss -35 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &
sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-25__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_60__ucg_False__cg_0.5__0 --gpu-devices 1 --criterion singular_htri  --switch-loss -25 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &

sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-15__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_60__ucg_False__cg_0.5__0 --gpu-devices 4 --criterion singular_htri  --switch-loss -15 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &

sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 70 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-35__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_70__ucg_False__cg_0.5__0 --gpu-devices 5 --criterion singular_htri  --switch-loss -35 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &
sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 70 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-25__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_70__ucg_False__cg_0.5__0 --gpu-devices 6 --criterion singular_htri  --switch-loss -25 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &

sleep 240m

sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 70 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-15__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_70__ucg_False__cg_0.5__0 --gpu-devices 0 --criterion singular_htri  --switch-loss -15 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &

sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-35__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_80__ucg_False__cg_0.5__0 --gpu-devices 1 --criterion singular_htri  --switch-loss -35 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &
sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-25__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_80__ucg_False__cg_0.5__0 --gpu-devices 4 --criterion singular_htri  --switch-loss -25 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &

sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 3 --optim adam --label-smooth --lr 0.0003 --max-epoch 80 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir feb_resnet_log/resnet50_sf_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-15__fcl_False__reg_svmo__dropout_incr__dau_crop,random-erase__pp_after__size_384__ep_80__ucg_False__cg_0.5__0 --gpu-devices 5 --criterion singular_htri  --switch-loss -15 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position after  --clip-grad 0.5 &
