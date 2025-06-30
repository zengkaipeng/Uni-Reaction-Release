import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json
from utils import ALL_COLS, COL_SHORT, validate

'''
usage: python draw_log.py --input /path/to/log/dir; 将所有需要对比的log_dir放在该目录下(/path/to/log/dir)
Output: plots of validation and test metrics, and train loss over epochs in the given log directory.
'''


def get_curve_tag(args, cols):
    name = ';'.join([f"{COL_SHORT.get(col, col)}={args[col]}" for col in cols if col in args])
    return name

def filter_args(all_log_args, cols):
    """
    returns the cols of the same value in all logs, and the cols of the diff value in all logs
    """
    shared_cols = []
    diff_cols = []

    all_col_values = {
        col: set() for col in cols if col in all_log_args[0]
    }
    for log_args in all_log_args:
        for col in log_args.keys():
            if col in all_col_values:
                all_col_values[col].add(log_args[col])

    for col in all_col_values.keys():
        if len(all_col_values[col]) == 1:
            shared_cols.append(col)
        else:
            diff_cols.append(col)
    return shared_cols, diff_cols

def validate_same_datapath(all_datapath):
    if len(set(all_datapath)) != 1:
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Args of the script: draw_compare_log')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input log dir')
    parser.add_argument('--cols', '-c', type=str, default='', help='using which cols for title, split by space, default is all cols')
    parser.add_argument('--test_only', '-t', action='store_true', help='whether to plot test metrics')
    parser.add_argument('--val_only', '-v', action='store_true', help='whether to plot valid metrics')
    parser.add_argument('--duplicate', '-d', action='store_true', help='if allow duplicate')
    args = parser.parse_args()
    input_dir = args.input
    test_only = args.test_only
    val_only = args.val_only
    do_not_draw_test = val_only and not test_only
    do_not_draw_val = test_only and not val_only

    all_valid_log = []

    for root, dirs, files in os.walk(input_dir):
        if len(dirs) == 0:
            log = validate(root)
            if log is not None:
                all_valid_log.append(log)
            else:
                continue

    if len(all_valid_log) == 0:
        print(f"No valid log found in {input_dir}.")
        exit(0)
    
    if validate_same_datapath([log['args']['data_path'] for log in all_valid_log]) is False:
        print("All logs should have the same data_path.")
        exit(0)
    
    if not args.duplicate:
        # check if there are duplicate logs
        all_mol_tag = set()
        filter_dup = []
        for log in all_valid_log:
            mol_tag = ';'.join([str(log['args'].get(col, ' ')) for col in ALL_COLS ])
            if mol_tag in all_mol_tag:
                print(f"Duplicate log found: {mol_tag}. removing it.")
                continue
            all_mol_tag.add(mol_tag)
            filter_dup.append(log)
        all_valid_log = filter_dup


    name_cols = args.cols.split(' ') if args.cols else ALL_COLS
    shared_cols, diff_cols = filter_args([log['args'] for log in all_valid_log], name_cols)

    title = get_curve_tag(all_valid_log[0]['args'], shared_cols)

    all_metric = list(all_valid_log[0]['valid_metric'][0].keys())

    # colors = plt.cm.viridis(np.linspace(0, 1, len(all_valid_log)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_valid_log))) 

    for metric in all_metric:
    # plot valid and test metrics
        plt.figure(figsize=(12, 6))
        for exp_idx, log in enumerate(all_valid_log):
            valid_metric = log['valid_metric']
            test_metric = log['test_metric']
            epochs = np.arange(len(valid_metric))
            legend_title = get_curve_tag(log['args'], diff_cols)
            if not do_not_draw_val:
                plt.plot(epochs, [x[metric] for x in valid_metric], 
                         label=legend_title+'(Val)', color=colors[exp_idx],
                         linestyle='-')
            if not do_not_draw_test:
                plt.plot(epochs, [x[metric] for x in test_metric], 
                         label=legend_title+'(Test)', color=colors[exp_idx],
                         linestyle='--')
            
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} of {title}')

        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(input_dir, f'{metric}_{title}.png'))
        plt.close()
    
    # plot loss
    plt.figure(figsize=(12, 6))
    for exp_idx, log in enumerate(all_valid_log):
        epochs = np.arange(len(log['train_loss']))
        legend_title = get_curve_tag(log['args'], diff_cols)
        plt.plot(epochs, log['train_loss'], label=legend_title, color=colors[exp_idx])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over epochs of ' + title)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(input_dir, f'train_loss_{title}.png'))
    plt.close()

print(f"Plots saved in the {input_dir} directory.")
