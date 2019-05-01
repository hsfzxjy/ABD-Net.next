import os
import glob
import os.path as osp
import argparse


def get_param(input_fn, layer, args):

    _, dir, name = input_fn.split('/')
    os.makedirs(osp.join(args.path, 'layer_' + str(layer), dir, name), 0o777, True)
    path = osp.join(args.path, dir, name)
    a = ['-i', osp.abspath(input_fn), '-p', osp.abspath(path), '-l', str(layer), '-w', args.model, '-a', args.arch, '--iter', str(args.iter)]
    print(a)
    # _, dir, name = input_fn.split('/')
    # os.makedirs(osp.join('without_orth_vis_output', dir, name), 0o777, True)
    # path = osp.join('without_orth_vis_output', dir, name)
    # b = ['-i', osp.abspath(input_fn), '-p', osp.abspath(path), '-l', str(layer), '-w', '../0111_cluster_pre_log/densenet121_fc512_fd_none_head_dan_none_head_xent_60_10_size_256__1/checkpoint_ep60.pth.tar']

    return [a, ]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('path')
    parser.add_argument('model')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--iter', type=int, default=50)
    options = parser.parse_args()

    params = []
    # for fn in glob.glob('vis_input/**/*'):
    #     params.append(get_param(fn, 5))
    for fn in glob.glob('vis_input/**/*'):
        params.extend(get_param(fn, options.layer, options))
    # for fn in glob.glob('vis_input/**/*'):
    #     params.extend(get_param(fn, 5))

    splitted = [params[i:i + 5] for i in range(0, len(params), 5)]
    import subprocess

    for group in splitted:
        ps = []
        for param in group:
            p = subprocess.Popen([
                'python', 'vis.py',
                *param
            ])
            ps.append(p)
        for p in ps:
            p.communicate()
