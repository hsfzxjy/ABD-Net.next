#!/usr/bin/env python3

import GPUtil
import subprocess
import argparse
import sys
import os
import json
from typing import Optional

'''
{
    "env": {},
    "arch": str,
    "epoch": str,
    "log_dir": str,
    "criterion": str,
    "fixbase": str,
    "fix_custom_loss": bool,
    "switch_loss": int
}
'''


def validate_arguments(args: dict) -> dict:

    def identity(x):
        return x

    validators = [
        ('env', dict, lambda d: {str(k): str(v) for k, v in d.items()}),
        ('arch', str, str),
        ('epoch', int, str),
        ('log_dir', str, str),
        ('criterion', str, str),
        ('fixbase', int, str),
        ('fix_custom_loss', bool, identity),
        ('switch_loss', int, lambda x: str(x) if x else ''),
    ]

    for name, type_, validator in validators:
        if not isinstance(args.get(name), type_):
            raise TypeError(
                f'Failed at {name} in {args}.'
            )

        args[name] = validator(args.get(name))

    return args


def get_arguments(filename: str):

    if filename == '-':
        fd = sys.stdin
    else:
        fd = open(filename, 'r')

    result = json.load(fd)
    fd.close()

    return result


def run_task(args: dict, gpu_id: int, dry_run: bool=False) -> Optional[subprocess.Popen]:

    cmd = [
        'python', 'train.py',

        '--root', 'data',
        '-s', 'market1501',
        '-t', 'market1501', 'cuhk03', 'dukemtmcreid',

        '-j', '4',
        '--height', '256', '--width', '128',

        '--optim', 'adam',
        '--lr', '0.0003',
        '--max-epoch', args['epoch'],
        '--stepsize', '20', '40',
        '--fixbase-epoch', args['fixbase'],
        '--open-layers', 'classifier', 'fc',
        '--train-batch-size', '32',
        '--test-batch-size', '100',

        '-a', args['arch'],
        '--save-dir', args['log_dir'],

        *(
            ['--switch-loss', args['switch_loss']]
            if args['switch_loss']
            else []
        ),
        '--criterion', args['criterion'],
        '--fix-custom-loss' if args['fix_custom_loss'] else '',
        '--label-smooth',

        '--gpu-devices', str(gpu_id),
    ]

    if dry_run:
        return [cmd, args['env']]

    if os.path.isdir(args['log_dir']):
        return None

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**args['env'], **os.environ}
    )


def get_next_available_gpu() -> Optional[int]:

    gpus = GPUtil.getAvailable(maxMemory=0.6)
    if not gpus:
        return None
    return gpus[0]


import time
import collections


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('input')
    argparser.add_argument('--dry-run', action='store_true', default=False)
    parsed = argparser.parse_args()

    arg_list = get_arguments(parsed.input)
    arg_list = collections.deque([
        validate_arguments(args) for args in arg_list
    ])

    if parsed.dry_run:
        results = [run_task(args, 0, True) for args in arg_list]
        print(json.dumps(results, indent=2))
        sys.exit(0)

    processes = []

    try:
        while arg_list:

            for process, args in processes:
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    processes.remove((process, args))

                    if process.returncode != 0:
                        print(f'FAILED: {args["log_dir"]}')

            gpu = get_next_available_gpu()
            if gpu is None:
                time.sleep(1)
                continue

            args = arg_list.popleft()
            process = run_task(args, gpu)
            if process is None:
                print(f'WARNING: Task {args["log_dir"]} exists. Skipped.')
            processes.append((process, args))
    except KeyboardInterrupt:
        for process, _ in processes:
            process.terminate()
            process.communicate()

        sys.exit()


if __name__ == '__main__':
    main()
