sleep 100m

ps aux | grep '0,3,4,5' | awk '{system("kill " $2)}'

sleep 10

reg_const=1 flip_eval=1 dropout_reduction=1 sing_beta=5e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501  -j 4 --height 384 --width 128 --eval-freq 1 --optim adam --label-smooth --lr 0.0003 --max-epoch 200 --stepsize 20 50  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 128 --test-batch-size 100 -a resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead --save-dir 0218_trick_log/resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead__crit_singular__htri_sb_5e-7__b_1e-3__sl_-175__fcl_False__reg_const_svmo__dropout_incr__dau_crop,random-erase__pp_before,after,cam,pam,layer5__size_384__ep_200__lh_.1__dr__b128__0 --gpu-devices 0,3,4,5 --criterion singular_htri  --switch-loss -175 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position before,after,cam,pam,layer5  --lambda-htri 0.1 &

sleep 300m


ps aux | grep '0,3,4,5' | awk '{system("kill " $2)}'
sleep 10

reg_const=1 flip_eval=1 dropout_reduction=1 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501  -j 4 --height 384 --width 128 --eval-freq 1 --optim adam --label-smooth --lr 0.0003 --max-epoch 200 --stepsize 20 50  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 64 --test-batch-size 100 -a resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead --save-dir 0218_trick_log/resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead__crit_singular__htri_sb_1e-6__b_1e-3__sl_-175__fcl_False__reg_const_svmo__dropout_incr__dau_crop,random-erase__pp_before,after,cam,pam,layer5__size_384__ep_200__lh_.1__dr__b64__0 --gpu-devices 0,3 --criterion singular_htri  --switch-loss -175 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position before,after,cam,pam,layer5  --lambda-htri 0.1 &

reg_const=1 flip_eval=1 dropout_reduction=1 sing_beta=1e-7 beta=1e-3 nohup python train_reg_crit.py --root data -s market1501 -t market1501  -j 4 --height 384 --width 128 --eval-freq 1 --optim adam --label-smooth --lr 0.0003 --max-epoch 200 --stepsize 20 50  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 64 --test-batch-size 100 -a resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead --save-dir 0218_trick_log/resnet50_sf_tr6_fc1024_fd_ab_nohead_dan_cam_pam_nohead__crit_singular__htri_sb_1e-7__b_1e-3__sl_-175__fcl_False__reg_const_svmo__dropout_incr__dau_crop,random-erase__pp_before,after,cam,pam,layer5__size_384__ep_200__lh_.1__dr__b64__0 --gpu-devices 4,5 --criterion singular_htri  --switch-loss -175 --regularizer svmo --dropout incr --data-augment crop,random-erase --penalty-position before,after,cam,pam,layer5  --lambda-htri 0.1 &
