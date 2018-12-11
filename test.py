import argparse

parser = argparse.ArgumentParser()
parser.add_argument('arch')
parser.add_argument('ckpt')
parser.add_argument('gpu')

parsed = parser.parse_args()

import subprocess
subprocess.Popen(
    [
        'python', 'train_imgreid_xent.py',
        '-s', 'market1501',
        '-t', 'market1501',
        '--height', '256',
        '--width', '128',
        '--test-batch-size', '100',
        '--evaluate',
        '-a', parsed.arch,
        '--load-weights', parsed.ckpt,
        '--save-dir', '../__',
        '--gpu-devices', parsed.gpu
    ]).communicate()
