import time
import os
import os.path as osp

def create_eval_dir(root, name='default'):
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{name}_{logtime}'
    logdir = osp.join(root, logdir)
    os.makedirs(logdir, exist_ok=True)
    return logdir