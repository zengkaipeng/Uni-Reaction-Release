import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json
import pandas as pd
from utils import COL_SHORT
# r_value
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
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
    parser.add_argument('--part', '-p', type=str, default='', help='use result of pred_{part}.csv')
    args = parser.parse_args()
    input_dir = args.input

    pred_csv = os.path.join(input_dir, f'pred_{args.part}_ralign.csv')

    data = pd.read_csv(pred_csv)
    true_yield = data['true_yield']
    pred_yield = data['pred_yield']
    
    # scater plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_yield, pred_yield, alpha=0.5, label='RAlign', color='pink')
    ralign_r2 = r2_score(true_yield, pred_yield)
    ralign_mse = mean_squared_error(true_yield, pred_yield)
    ralign_mae = mean_absolute_error(true_yield, pred_yield)
    ralign_r = pearsonr(true_yield, pred_yield)[0]
    # plt.plot([true_yield.min(), true_yield.max()], [true_yield.min(), true_yield.max()], color='red', linestyle='--')
    # 画出拟合线
    z = np.polyfit(true_yield, pred_yield, 1)
    p = np.poly1d(z)
    plt.plot(true_yield, p(true_yield), color='red', linestyle='--', label=f'ralign')


    t5_pred = pd.read_csv(os.path.join(input_dir, f'pred_{args.part}_t5chem.csv'))
    t5_true_yield = t5_pred['target'] * 100
    t5_pred_yield = t5_pred['prediction'] * 100
    plt.scatter(t5_true_yield, t5_pred_yield, alpha=0.5, label='T5', color='green')
    t5_z = np.polyfit(t5_true_yield, t5_pred_yield, 1)
    t5_p = np.poly1d(t5_z)
    t5_r2 = r2_score(t5_true_yield, t5_pred_yield)
    t5_mse = mean_squared_error(t5_true_yield, t5_pred_yield)
    t5_mae = mean_absolute_error(t5_true_yield, t5_pred_yield)
    t5_r = pearsonr(t5_true_yield, t5_pred_yield)[0]

    plt.plot(t5_true_yield, t5_p(t5_true_yield), color='blue', linestyle='--', label='T5 Fit Line')

    # fix the axis limits
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)

    # plot y = x
    plt.plot([0, 100], [0, 100], color='black', linestyle='--', label='y=x')

    plt.title(f'SM Comparison: {args.part}\n metric: ours/T5chem; R2: {ralign_r2:.2f} / {t5_r2:.2f}; MSE: {ralign_mse:.2f} / {t5_mse:.2f}; MAE : {ralign_mae:.2f} / {t5_mae:.2f}; r**2 : {ralign_r**2:.2f} / {t5_r**2:.2f}')

    plt.legend()
    # plt.xlim(true_yield.min() - 0.1, true_yield.max() + 0.1)
    # plt.ylim(true_yield.min() - 0.1, true_yield.max() + 0.1)
    # plt.xticks(np.arange(true_yield.min() - 0.1, true_yield.max() + 0.1, 0.1))
    # plt.yticks(np.arange(true_yield.min() - 0.1, true_yield.max() + 0.1, 0.1))  
    plt.xlabel('True Yield')
    plt.ylabel('Predicted Yield')
    plt.grid()
    plt.savefig(os.path.join(input_dir, f'scatter_{args.part}.png'))
    plt.close()

print(f"Plots saved in the {input_dir} directory.")
