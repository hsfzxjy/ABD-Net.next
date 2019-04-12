for ckpt in ../cvpr2019/log/fat_64_0.1/*.pth.tar; do
    flip_eval=1 sing_beta=1e-6 beta=1e-3 head_col=2 part_num=2  python eval_acc.py -s market1501 -t market1501 --eval-freq 5 --label-smooth --criterion singular_htri --lambda-htri 0.1  --dropout incr --data-augment crop,random-erase --regularizer svmo --margin 1.2 --train-batch-size 64 --switch-loss -178 --penalty-position before,after,cam,pam,layer5 --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 200 --save-dir ../__0408_aicity_log/1e-6_1e-3_final --arch resnet50_sf_abd_fc1024_fd_abc_nohead_dan_cam_pam_nohead --gpu-devices 1,4 --evaluate --load-weights $ckpt
done

for ckpt in ../cvpr2019/log/fat_64_0.01/*.pth.tar; do
    flip_eval=1 sing_beta=1e-6 beta=1e-3 head_col=2 part_num=2  python eval_acc.py -s market1501 -t market1501 --eval-freq 5 --label-smooth --criterion singular_htri --lambda-htri 0.1  --dropout incr --data-augment crop,random-erase --regularizer svmo --margin 1.2 --train-batch-size 64 --switch-loss -178 --penalty-position before,after,cam,pam,layer5 --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 200 --save-dir ../__0408_aicity_log/1e-6_1e-3_final --arch resnet50_sf_abd_fc1024_fd_abc_nohead_dan_cam_pam_nohead --gpu-devices 1,4 --evaluate --load-weights $ckpt
done

for ckpt in ../cvpr2019/log/fat_64_0.05/*.pth.tar; do
    flip_eval=1 sing_beta=1e-6 beta=1e-3 head_col=2 part_num=2  python eval_acc.py -s market1501 -t market1501 --eval-freq 5 --label-smooth --criterion singular_htri --lambda-htri 0.1  --dropout incr --data-augment crop,random-erase --regularizer svmo --margin 1.2 --train-batch-size 64 --switch-loss -178 --penalty-position before,after,cam,pam,layer5 --height 384 --width 128 --fixbase-epoch 10 --optim adam --lr 0.0003 --stepsize 20 40 --max-epoch 200 --save-dir ../__0408_aicity_log/1e-6_1e-3_final --arch resnet50_sf_abd_fc1024_fd_abc_nohead_dan_cam_pam_nohead --gpu-devices 1,4 --evaluate --load-weights $ckpt
done
