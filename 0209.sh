sleep 240m

reg_const=1 sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height 384 --width 128 --eval-freq 1 --optim adam --label-smooth --lr 0.0003 --max-epoch 100 --stepsize 20 40  --open-layers classifier fc reduction classifier2 --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a resnet50_sf_tr1_fc512_fd_ab_nohead_dan_cam_pam_nohead --save-dir 0208_trick_log/tr1/resnet50_sf_tr1_fc512_fd_ab_nohead_dan_cam_pam_nohead__crit_singular_htri__sb_1e-7__b_1e-3__sl_-75__fcl_False__reg_svmo_const__dropout_incr__dau_crop,color-jitter,random-erase__pp_before,after,cam,pam__size_384__ep_100__ucg_False__cg_0.5__0 --gpu-devices 7 --criterion singular_htri  --switch-loss -75 --regularizer svmo --dropout incr --data-augment crop,color-jitter,random-erase --penalty-position before,after,cam,pam  --clip-grad 0.5 &


