from glob import glob
import os
import subprocess
import re


def get_exps():
    return glob('./log/*')


def get_exp(dir):
    """
    returns (name,DAN_part, tarname, last_log)
    """
    # basename = dir.rstrip('/').split('/')[-1]
    if 'densenet_DAN_fc512' in dir:
        name = 'densenet121_DAN_fc512'
    else:
        name = 'densenet121_DAN'
    matched = re.findall(r'(_[spc]_)', dir)
    if matched:
        part = matched[0][1]
    else:
        part = ''
    tarname = dir + '/checkpoint_ep60.pth.tar'
    last_log = subprocess.check_output(['tail', '-n', '11', dir + '/log_train.txt']).decode()

    return name, part, tarname, last_log


def run_exp(dir):

    name, part, tarname, last_log = get_exp(dir)

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
    print('name', dir)
    print('log')
    print(last_log)
    print('result')
    print(result)
    print('----')


for x in get_exps():
    run_exp(x)
