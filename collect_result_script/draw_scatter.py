import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json
import pandas as pd
from utils import COL_SHORT
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
    parser.add_argument('--part', '-p', type=str, default='test', help='use result of pred_{part}.csv')
    args = parser.parse_args()
    input_dir = args.input

    log_json = os.path.join(input_dir, 'log.json')
    if not os.path.exists(log_json):
        print(f"Log file {log_json} does not exist.")
        exit(0)
    
    with open(log_json, 'r') as f:
        log = json.load(f)
    
    name_cols = ALL_COLS
    name = get_curve_tag(log['args'], name_cols)
    # plot valid and test metrics
    pred_csv = os.path.join(input_dir, f'pred_{args.part}.csv')
    if not os.path.exists(pred_csv):
        print(f"Prediction file {pred_csv} does not exist.")
        exit(0)

    data = pd.read_csv(pred_csv)
    true_yield = data['label']
    pred_yield = data['prediction']

    r2, mae, mse = r2_score(true_yield, pred_yield), mean_absolute_error(true_yield, pred_yield), mean_squared_error(true_yield, pred_yield)

    # scater plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_yield, pred_yield, alpha=0.5)
    # plt.plot([true_yield.min(), true_yield.max()], [true_yield.min(), true_yield.max()], color='red', linestyle='--')
    # 画出拟合线
    z = np.polyfit(true_yield, pred_yield, 1)
    p = np.poly1d(z)
    plt.plot(true_yield, p(true_yield), color='red', linestyle='--', label='Fit Line')
    plt.legend()
    # plt.xlim(true_yield.min() - 0.1, true_yield.max() + 0.1)
    # plt.ylim(true_yield.min() - 0.1, true_yield.max() + 0.1)
    # plt.xticks(np.arange(true_yield.min() - 0.1, true_yield.max() + 0.1, 0.1))
    # plt.yticks(np.arange(true_yield.min() - 0.1, true_yield.max() + 0.1, 0.1))  
    plt.xlabel('True Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'{name}\nR2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')
    plt.grid()
    plt.savefig(os.path.join(input_dir, f'scatter_{args.part}.png'))
    plt.close()

print(f"Plots saved in the {input_dir} directory.")
