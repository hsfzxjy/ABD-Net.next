sleep 120m

ps aux | grep 'ces 3,4' | awk '{system("kill " $2)}'
ps aux | grep 'ces 0,2' | awk '{system("kill " $2)}'

sleep 5

flip_eval=1 sing_beta=8e-7 beta=1e-3 head_col=2 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 1 --label-smooth --criterion singular_htri --lambda-htri 0.1  --dropout incr --data-augment crop,random-erase --regularizer svmo --margin 1.2 --train-batch-size 64 --switch-loss -178 --penalty-position before,after,cam,pam,layer5 --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 200 --save-dir log/duke_8e-7_1e-3_with_abc --arch resnet50_sf_abd_fc1024_fd_abc_nohead_dan_cam_pam_nohead --gpu-devices 3,4 &

flip_eval=1 sing_beta=2e-7 beta=1e-3 head_col=2 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 1 --label-smooth --criterion singular_htri --lambda-htri 0.1  --dropout incr --data-augment crop,random-erase --regularizer svmo --margin 1.2 --train-batch-size 64 --switch-loss -178 --penalty-position before,after,cam,pam,layer5 --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 200 --save-dir log/duke_2e-7_1e-3_with_abc --arch resnet50_sf_abd_fc1024_fd_abc_nohead_dan_cam_pam_nohead --gpu-devices 0,2 &
