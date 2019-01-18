import os
import glob
import os.path as osp


def get_param(input_fn, layer):

    _, dir, name = input_fn.split('/')
    os.makedirs(osp.join('with_orth_vis_output', dir, name), 0o777, True)
    path = osp.join('with_orth_vis_output', dir, name)
    a = ['-i', osp.abspath(input_fn), '-p', osp.abspath(path), '-l', str(layer), '-w', '../log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b___sl_0__fcl_False__reg_none__dropout_none__dau_crop__pp_before__size_256__0//checkpoint_ep60.pth.tar']
    # _, dir, name = input_fn.split('/')
    # os.makedirs(osp.join('without_orth_vis_output', dir, name), 0o777, True)
    # path = osp.join('without_orth_vis_output', dir, name)
    # b = ['-i', osp.abspath(input_fn), '-p', osp.abspath(path), '-l', str(layer), '-w', '../0111_cluster_pre_log/densenet121_fc512_fd_none_head_dan_none_head_xent_60_10_size_256__1/checkpoint_ep60.pth.tar']

    return [a, ]


if __name__ == '__main__':

    params = []
    # for fn in glob.glob('vis_input/**/*'):
    #     params.append(get_param(fn, 5))
    for fn in glob.glob('vis_input/**/*'):
        params.extend(get_param(fn, 4))
        params.extend(get_param(fn, 5))

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
