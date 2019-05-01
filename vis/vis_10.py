import os
import os.path as osp

CD = osp.abspath(osp.dirname(__file__))

for x in range(10):

    os.system(f'python3 vis.py -i {CD}/pic/{x}.jpg -p {CD}/generated/{x}.jpg/ -l 4')
