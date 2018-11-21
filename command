python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_pam_fc512 --save-dir log/densenet121_dan_fc512-market-xent --gpu-devices 0 --criterion xent

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_pam_fc512 --save-dir log/eval-densenet121_dan_fc512-market-xent --gpu-devices 0 --criterion xent --evaluate --load-weights log/densenet121_dan_fc512-market-xent/checkpoint_ep60.pth.tar

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_pam --save-dir log/densenet121_pam_fc512-market-xent --gpu-devices 0 --criterion xent

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121 --save-dir ../log_find_densenet/dprn_densenet_amsgrad_60_10_xent --gpu-devices 5 --criterion xent

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_amsgrad_60_10_xent --gpu-devices 6 --criterion xent

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_amsgrad_60_0_xent --gpu-devices 7 --criterion xent

python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121 --save-dir ../log_find_densenet/dprn_densenet_amsgrad_60_0_xent --gpu-devices 5 --criterion xent


python train_imgreid_xent_htri.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_amsgrad_60_0_xent_htri --gpu-devices 7

python train_imgreid_xent_htri.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_adam_60_0_xent_htri --gpu-devices 6

python train_imgreid_xent_htri.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_adam_60_10_xent_htri --gpu-devices 5

python train_imgreid_xent_htri.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_fc512 --save-dir ../log_find_densenet/dprn_densenet_fc512_amsgrad_60_10_xent_htri --gpu-devices 3


DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_s_amsgrad_60_10_xent --gpu-devices 7 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_p_amsgrad_60_10_xent --gpu-devices 6 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_c_amsgrad_60_10_xent --gpu-devices 5 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_c_amsgrad_60_10_xent --gpu-devices 4 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_p_amsgrad_60_10_xent --gpu-devices 3 &

DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_s_amsgrad_60_10_xent --gpu-devices 2 &



DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_s_amsgrad_60_0_xent --gpu-devices 7 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_p_amsgrad_60_0_xent --gpu-devices 6 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_fc512 --save-dir log/dprn_densenet_DAN_fc512_c_amsgrad_60_0_xent --gpu-devices 5 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_c_amsgrad_60_0_xent --gpu-devices 4 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_p_amsgrad_60_0_xent --gpu-devices 3 &

DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 0 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN --save-dir log/dprn_densenet_DAN_s_amsgrad_60_0_xent --gpu-devices 2 &


DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_s_amsgrad_60_10_xent --gpu-devices 7 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_p_amsgrad_60_10_xent --gpu-devices 6 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_c_amsgrad_60_10_xent --gpu-devices 5 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_c_amsgrad_60_10_xent --gpu-devices 4 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_p_amsgrad_60_10_xent --gpu-devices 3 &

DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_s_amsgrad_60_10_xent --gpu-devices 2 &



DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_s__dan_open__amsgrad_60_10_xent --gpu-devices 7 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_p__dan_open__amsgrad_60_10_xent --gpu-devices 6 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat_fc512 --save-dir log/dprn_densenet_DAN_cat_fc512_c__dan_open__amsgrad_60_10_xent --gpu-devices 5 &

DAN_part=c nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_c__dan_open__amsgrad_60_10_xent --gpu-devices 4 &

DAN_part=p nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_p__dan_open__amsgrad_60_10_xent --gpu-devices 3 &

DAN_part=s nohup python train_imgreid_xent.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_DAN_cat_s__dan_open__amsgrad_60_10_xent --gpu-devices 2 &



python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim amsgrad --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc danet_head --train-batch-size 32 --test-batch-size 100 -a densenet121_DAN_cat --save-dir log/dprn_densenet_amsgrad_60_10_xent --gpu-devices 2

