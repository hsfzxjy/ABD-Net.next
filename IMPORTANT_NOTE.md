## New Training Script

To get a better result, use

```bash
python train.py \
--root data \
-s market1501 \
-t market1501 \
-j 4 \
--height 256 \
--width 128 \
--optim amsgrad \
--label-smooth \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--fixbase-epoch 10 \
--open-layers classifier fc \
--train-batch-size 32 \
--test-batch-size 100 \
-a densenet121_fc512 \
--save-dir log/densenet121_fc512-market-xent \
--gpu-devices 0 \
--criterion CRITERION
```

where `CRITERION` may be `lowrank` or `singular`.

envvar `beta=BETA` must be set for balancing.

For `lowrank`, `BETA` is about `1e-10 ~ 1e-11`.

For `singular`, envvar `use_log=1` may be set to control whether to use logarithm or not. If set, `BETA` is about `1e-2`, and `1e-5 ~ 1e-6` otherwise.

## Evaluate on Valset

Extract `valset` under `data` to form the structure as below:

```
deep-person-reid/
 | data/
 | valset/
   | valSet/
     | gallery/
     | query/
```

Change argument `-t` to `valset` to evaluate.
