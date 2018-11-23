import os
import glob
import os.path as osp


def run_task(input_fn):

    _, dir, name = input_fn.split('/')
    os.makedirs(osp.join('vis_output', dir, name), 0o777, True)
    path = osp.join('vis_output', dir, name)
    print(input_fn, path)


if __name__ == '__main__':

    for fn in glob.glob('vis_input/**/*'):
        run_task(fn)
