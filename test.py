from glob import glob
import os
import subprocess
import re


import argparse


def get_exps():
    p = argparse.ArgumentParser()
    p.add_argument('glob', default='./log/*')
    return glob(p.parse_args().glob)


def get_exp(dir):
    """
    returns (name,DAN_part, tarname, last_log)
    """
    # basename = dir.rstrip('/').split('/')[-1]
    if 'densenet_DAN_fc512' in dir:
        name = 'densenet121_DAN_fc512'
    elif 'densenet_DAN_cat_fc512' in dir:
        name = 'densenet121_DAN_cat_fc512'
    elif 'densenet_DAN_cat' in dir:
        name = 'densenet121_DAN_cat'
    elif 'densenet_DAN' in dir:
        name = 'densenet121_DAN'
    elif 'densenet_CAM_cl_cat_fc512' in dir:
        name = 'densenet121_CAM_cl_cat_fc512'
    elif 'densenet_CAM_noncl_cat_fc512' in dir:
        name = 'densenet121_CAM_noncl_cat_fc512'
    elif 'densenet_CAM_cat' in dir:
        name = 'densenet121_CAM_cat'
    elif 'densenet_fc512' in dir:
        name = 'densenet121_fc512'
    elif 'densenet_cl_sum_fc512' in dir:
        name = 'densenet121_cl_sum_fc512'
    elif 'densenet_cl_sum' in dir:
        name = 'densenet121_cl_sum'
    elif 'densenet_cl_fc512' in dir:
        name = 'densenet121_cl_fc512'
    elif 'densenet_cl' in dir:
        name = 'densenet121_cl'
    elif 'densenet' in dir:
        name = 'densenet121'
    matched = re.findall(r'(_[spc]_)', dir)
    if matched:
        part = matched[0][1]
    else:
        part = ''
    epch = 100
    while not os.path.exists(dir + '/checkpoint_ep{}.pth.tar'.format(epch)) and epch > 0:
        epch -= 1
    if epch == 0:
        return
    tarname = dir + '/checkpoint_ep{}.pth.tar'.format(epch)
    last_log = subprocess.check_output(['tail', '-n', '11', dir + '/log_train.txt']).decode()

    return name, part, tarname, last_log


def run_exp(dir):

    x = get_exp(dir)
    if x is None:
        return
    name, part, tarname, last_log = x

    p = subprocess.Popen(
        [
            'python', 'eval.py',
            # '--cd', '.',
            '--arch', name,
            '--snap_shot', tarname
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, 'DAN_part': part}
    )

    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(stderr.decode())
    else:
        result = stdout.decode().strip().split('\n')[-1]
        with open(os.path.join(dir, 'eval.txt'), 'w') as f:
            f.write(result)

    print('----')
    print('arch', name, 'part', part)
    print('name', dir)
    print('log')
    print(last_log)
    print('result')
    print(result)
    print('----')


for x in get_exps():
    run_exp(x)
