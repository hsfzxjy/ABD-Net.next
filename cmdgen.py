#!/usr/bin/env python3

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', default='none_nohead', choices=['%s_%s' % (i, j) for i in ['ab_c', 'ab', 'a', 'none'] for j in ['head', 'nohead']])
    parser.add_argument('--dan', default='none_nohead', choices=['%s_%s' % (i, j) for i in ['cam_pam', 'cam', 'pam', 'none'] for j in ['head', 'nohead']])
    parser.add_argument('--arch', default='densenet121_fc512')
    parser.add_argument('--size', default='256', choices=['256', '384'])
    parser.add_argument('--pp', type=str, default='before', choices=['before', 'pam', 'cam', 'pam,cam', 'before,pam', 'before,cam', 'before,pam,cam', 'layer5'], help='penalty position')
    parser.add_argument('--reg', default='none', choices=['none', 'so', 'svmo', 'svdo'])
    parser.add_argument('--dropout', type=str, default='none', choices=['none', 'incr', 'fix'])
    parser.add_argument('--dau', type=str, choices=['none', 'crop', 'random-erase', 'color-jitter', 'crop,random-erase', 'crop,color-jitter'], default='crop')
    parser.add_argument('--crit', type=str, default='xent')

    parser.add_argument('--num', default=4, type=int)
    parser.add_argument('--gpu-start', default=0, type=int)
    parser.add_argument('--sing-beta', default='')
    parser.add_argument('--beta', default='')
    parser.add_argument('--sl', type=int, default=0)
    parser.add_argument('--fcl', action='store_true', default=False)
    parser.add_argument('--dir', default='log')
    parser.add_argument('--skip', type=str, default='')

    options = parser.parse_args()

    template = '''sing_beta={sing_beta} beta={beta} nohup python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height {size} --width 128 --optim adam --label-smooth --lr 0.0003 --max-epoch 60 --stepsize 20 40  --open-layers classifier fc --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a {model} --save-dir {dir}/{model}__crit_{crit}__sb_{sing_beta}__b_{beta}__sl_{sl}__fcl_{fcl}__reg_{reg}__dropout_{dropout}__dau_{dau}__pp_{pp}__size_{size}__{index} --gpu-devices {gpu} --criterion {crit} {fcl_command} --switch-loss {sl} --regularizer {reg} --dropout {dropout} --data-augment {dau} --penalty-position {pp} &'''

    gpu_start = options.gpu_start

    for index in range(options.num):

        gpu = (gpu_start + index) % 8

        while str(gpu) in options.skip.split(','):
            gpu_start += 1
            gpu = (gpu + 1) % 8

        args = {
            'sing_beta': options.sing_beta,
            'beta': options.beta,
            'size': options.size,
            'model': f'{options.arch}_fd_{options.fd}_dan_{options.dan}',
            'dir': options.dir,
            'crit': options.crit,
            'sl': options.sl,
            'fcl': options.fcl,
            'reg': options.reg,
            'dropout': options.dropout,
            'dau': options.dau,
            'pp': options.pp,
            'size': options.size,
            'index': index,
            'gpu': gpu,
            'fcl_command': '--fix-custom-loss' if options.fcl else '',
        }
        print(template.format(**args))
