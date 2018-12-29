import argparse

parser = argparse.ArgumentParser()
parser.add_argument('arch')
parser.add_argument('ckpt')
parser.add_argument('gpu')
parser.add_argument('-h', default='256')

parsed = parser.parse_args()

import subprocess
subprocess.Popen(
    [
        'python', 'train_reg_crit.py',
        '-s', 'market1501',
        '-t', 'market1501',
        '--height', parsed.h,
        '--width', '128',
        '--test-batch-size', '100',
        '--evaluate',
        '-a', parsed.arch,
        '--load-weights', parsed.ckpt,
        '--save-dir', '../__',
        '--gpu-devices', parsed.gpu
    ]).communicate()
