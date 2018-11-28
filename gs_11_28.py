import os
import subprocess

gpus = {2, 3, 4, 5, 6, 7}

betas = ['{}e-{}'.format(base, exp) for base in range(1, 10) for exp in [7, 8, 9, 6]]


def get_process(beta, gpu):

    return subprocess.Popen(['sh', '-c', f'echo {beta} {gpu} && sleep 2'])

    return subprocess.Popen(
        f'python train.py --root data -s market1501 -t market1501 -j 4 --height 256 --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 40 --stepsize 20 40 --fixbase-epoch 10 --open-layers classifier fc pa ca ca1 ca2 --train-batch-size 32 --test-batch-size 100 -a densenet121_CAM_noncl_cat_fc512 --save-dir finetune_log/dprn_densenet_CAM_noncl_cat_fc512__open__adam_40_10_singular_{beta} --criterion singular --gpu-devices {gpu}'.split(),
        env={**os.environ, 'beta': beta}
    )


processes = []

for beta in betas:

    while not gpus:
        for gpu, p in processes:
            try:
                p.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            else:
                processes.remove((gpu, p))
                gpus.add(gpu)

    gpu = gpus.pop()
    processes.append((gpu, get_process(beta, gpu)))
