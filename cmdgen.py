#!/usr/bin/env python3

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', default='none_nohead', choices=['%s_%s' % (i, j) for i in ['ab_c', 'ab', 'a', 'none'] for j in ['head', 'nohead']])
    parser.add_argument('--dan', default='none_nohead', choices=['%s_%s' % (i, j) for i in ['cam_pam', 'cam', 'pam', 'none'] for j in ['head', 'nohead']])
    parser.add_argument('--arch', default='densenet121_fc512')
    parser.add_argument('--size', default='256', choices=['256', '384'])
    parser.add_argument('--pp', type=str, default='before', choices=['before', 'pam', 'cam', 'pam,cam', 'before,pam', 'before,cam', 'before,pam,cam', 'layer5', 'after', 'all_layers', 'before,layer5', 'after,layer5', 'after,cam', 'before,after,cam,pam', 'before,before2,after,cam,pam', 'before,after,cam,pam,layer5'], help='penalty position')
    parser.add_argument('--reg', default='none', choices=['none', 'so', 'svmo', 'svdo'])
    parser.add_argument('--dropout', type=str, default='none', choices=['none', 'incr', 'fix'])
    parser.add_argument('--dau', type=str, choices=['none', 'crop', 'random-erase', 'color-jitter', 'crop,random-erase', 'crop,color-jitter', 'crop,color-jitter,random-erase'], default='crop')
    parser.add_argument('--crit', type=str, default='xent')

    parser.add_argument('--num', default=4, type=int)
    parser.add_argument('--gpu-start', default=0, type=int)
    parser.add_argument('--sing-beta', default='')
    parser.add_argument('--beta', default='')
    parser.add_argument('--sl', type=int, default=0)
    parser.add_argument('--fcl', action='store_true', default=False)
    parser.add_argument('--dir', default='log')
    parser.add_argument('--skip', type=str, default='')
    parser.add_argument('--epoch', type=str, default='60')
    parser.add_argument('--ucg', default=False, action='store_true')
    parser.add_argument('--cg', type=float, default=.5)
    parser.add_argument('--nonohup', action='store_true')
    parser.add_argument('--open', default='classifier fc')
    parser.add_argument('--eval-freq', default='3')

    options = parser.parse_args()

    template = '''sing_beta={sing_beta} beta={beta} {nohup_start} python train_reg_crit.py --root data -s market1501 -t market1501 -j 4 --height {size} --width 128 --eval-freq {eval_freq} --optim adam --label-smooth --lr 0.0003 --max-epoch {epoch} --stepsize 20 40  --open-layers {open} --fixbase-epoch 10  --train-batch-size 32 --test-batch-size 100 -a {model} --save-dir {dir}/{model}__crit_{crit}__sb_{sing_beta}__b_{beta}__sl_{sl}__fcl_{fcl}__reg_{reg}__dropout_{dropout}__dau_{dau}__pp_{pp}__size_{size}__ep_{epoch}__ucg_{ucg}__cg_{cg}__{index} --gpu-devices {gpu} --criterion {crit} {fcl_command} --switch-loss {sl} --regularizer {reg} --dropout {dropout} --data-augment {dau} --penalty-position {pp} {ucg_command} --clip-grad {cg} {nohup_end}'''

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
            'epoch': options.epoch,
            'fcl_command': '--fix-custom-loss' if options.fcl else '',
            'ucg': options.ucg,
            'cg': options.cg,
            'ucg_command': '--use-clip-grad' if options.ucg else '',
            'nohup_start': 'nohup' if not options.nonohup else '',
            'nohup_end': '&' if not options.nonohup else '',
            'eval_freq': options.eval_freq,
            'open': options.open
        }
        print(template.format(**args))
