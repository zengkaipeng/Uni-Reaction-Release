import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import json

'''
usage: python draw_log.py --input /path/to/log/dir
Output: plots of validation and test metrics, and train loss over epochs in the given log directory.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Args of the script: draw_log')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input log dir')
    args = parser.parse_args()
    input_dir = args.input

    log_json = os.path.join(input_dir, 'log.json')
    if not os.path.exists(log_json):
        print(f"Log file {log_json} does not exist.")
        exit(0)
    
    with open(log_json, 'r') as f:
        log = json.load(f)
    

    # plot valid and test metrics
    valid_metric = log['valid_metric']
    test_metric = log['test_metric']

    epochs = np.arange(len(valid_metric))
    for metric in valid_metric[0].keys():
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, [x[metric] for x in valid_metric], label='Validation ' + metric)
        plt.plot(epochs, [x[metric] for x in test_metric], label='Test ' + metric, linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} over epochs')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(input_dir, f'{metric}_over_epochs.png'))
        plt.close()
    
    # plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, log['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(input_dir, 'train_loss_over_epochs.png'))
    plt.close()

print(f"Plots saved in the {input_dir} directory.")
