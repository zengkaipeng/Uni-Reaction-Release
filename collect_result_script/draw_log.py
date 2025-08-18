import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json
from utils import COL_SHORT
'''
usage: python draw_log.py --input /path/to/log/dir
Output: plots of validation and test metrics, and train loss over epochs in the given log directory.
'''

ALL_COLS = ['dim', 'heads', 'n_layer', 'dropout', 'lr', 'use_temperature', 'use_volumn', 'use_solvent_volumn', 'volumn_norm', 'condition_both']

def get_curve_tag(args, cols):
    name = ';'.join([f"{COL_SHORT.get(col, col)}={args[col]}" for col in cols if col in args])
    return name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Args of the script: draw_log')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input log dir')
    parser.add_argument('--cols', '-c', type=str, default='', help='using which cols for title, split by space, default is all cols')
    parser.add_argument('--skip_first', '-sk', type=int, default=0, help='whether to skip the first epoch, default is 0')
    parser.add_argument('--notest', action='store_true', default=False, help='whether to skip the first epoch, default is 0')
    args = parser.parse_args()
    input_dir = args.input

    log_json = os.path.join(input_dir, 'log.json')
    if not os.path.exists(log_json):
        print(f"Log file {log_json} does not exist.")
        exit(0)
    
    with open(log_json, 'r') as f:
        log = json.load(f)
    
    name_cols = args.cols.split(' ') if args.cols else ALL_COLS
    name = get_curve_tag(log['args'], name_cols)
    # plot valid and test metrics
    valid_metric = log['valid_metric'][args.skip_first:]
    if not args.notest:
        test_metric = log['test_metric'][args.skip_first:]
    else:
        test_metric = None
    epochs = np.arange(len(valid_metric))
    for metric in valid_metric[0].keys():
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, [x[metric] for x in valid_metric], label='Validation ' + metric)
        if test_metric is not None:
            plt.plot(epochs, [x[metric] for x in test_metric], label='Test ' + metric, linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} of {name}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(input_dir, f'{metric}_over_epochs.png'))
        plt.close()
    
    # plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, log['train_loss'][args.skip_first:], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(input_dir, 'train_loss_over_epochs.png'))
    plt.close()

print(f"Plots saved in the {input_dir} directory.")
