import os
import json

ALL_COLS = ['dim', 'heads', 'n_layer', 'dropout', 'lr', 'use_temperature', 'use_volumn', 'use_solvent_volumn', 'volumn_norm', 'condition_both']

COL_SHORT = {
        'dim': 'd', 'heads': 'h', 'n_layer': 'l', 'dropout': 'dp',
        'lr': 'lr', 'use_temperature': 'T', 'use_volumn': 'V',
        'use_solvent_volumn': 'SV', 'volumn_norm': 'VN', 'condition_both': 'CB'
    }

SHORT2COL = {v: k for k, v in COL_SHORT.items()}

def validate(log_dir, notest=False):
    """
        当给定log目录不满足一下条件时，返回None
        1. log.json文件不存在
        2. log.json keys中没有'args', 'valid_metric', 'test_metric', 'train_loss'
        3. log.json 记录的epoch数与train_loss, valid_metric, test_metric的长度不一致
        否则返回：log.json文件的内容
    """
    log_json = os.path.join(log_dir, 'log.json')
    if not os.path.exists(log_json):
        return None
    with open(log_json, 'r') as f:
        log = json.load(f)
    if 'args' not in log or 'valid_metric' not in log or ('test_metric' not in log and not notest) or 'train_loss' not in log:
        return None

    sz = set([log['args']['epoch'], len(log['train_loss']), len(log['valid_metric'])])
    if not notest:
        sz.add(len(log['test_metric']))
    if len(sz) != 1:
        return None

    return log