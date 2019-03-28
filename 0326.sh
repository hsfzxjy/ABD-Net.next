part_num=1 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_dan_ow_of --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 6&

part_num=1 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_dan_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 1&

part_num=2 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_dan_p2 --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 2&

part_num=1 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_ow_of_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 3&

part_num=2 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_ow_of_p2 --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 4&

part_num=2 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s market1501 -t market1501 --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_market_log/baseline_p2_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 5&

sleep 240m

part_num=1 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_dan_ow_of --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 6&

part_num=1 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_dan_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 1&

part_num=2 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_dan_p2 --arch resnet50_abl_fc1024_fd_none_nohead_dan_cam_pam_nohead --gpu-devices 2&

part_num=1 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_ow_of_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 3&

part_num=2 sing_beta=1e-6 beta=1e-3 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion singular --lambda-htri 0.1  --dropout none --data-augment crop --regularizer svmo --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_ow_of_p2 --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 4&

part_num=2 sing_beta=1e-6 beta=1e-3 tb=1 nohup python train_reg_crit.py -s dukemtmcreid -t dukemtmcreid --eval-freq 5 --flip-eval --label-smooth --criterion xent --lambda-htri 0.1  --dropout none --data-augment crop --regularizer none --margin 1.2 --train-batch-size 32 --switch-loss -55 --penalty-position before,after,cam,pam --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 80 --save-dir abl_duke_log/baseline_p2_tb --arch resnet50_abl_fc1024_fd_none_nohead_dan_none_nohead --gpu-devices 5&