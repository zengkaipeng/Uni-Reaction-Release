#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp
import json
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Args of the script: collect_iid')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input dir')
    parser.add_argument('--best_by', '-b', type=str, default='R2', help='Metric to find best epoch')
    parser.add_argument('--ood_marker', '-ood', type=str, default='Test.*', help='Marker for OOD datasets')
    args = parser.parse_args()
    input_file = args.input
    ood_marker = args.ood_marker
    # os walking
    specific_args = ['dim', 'heads', 'n_layer', 'dropout', 'lr', 'condition_config', 'condition_both']
    # all_exps = ['FullCV_01', 'FullCV_02', 'FullCV_03', 'FullCV_04', 'FullCV_05',
    #             'FullCV_06', 'FullCV_07', 'FullCV_08', 'FullCV_09', 'FullCV_10']
    best_by = args.best_by
    all_task_result = {}
    for root, dirs, files in os.walk(input_file):
        if len(dirs) == 0 and len(files) == 3:
            with open(osp.join(root, 'log.json'), 'r') as f:
                log = json.load(f)
            args = log['args']
            exp_dataset = args['data_path'].split('/')[-1]
            if not re.match(ood_marker, exp_dataset):
                continue
            if exp_dataset not in all_task_result:
                all_task_result[exp_dataset] = {}
            all_model_result = all_task_result[exp_dataset]
            specific_args_val = [args[arg] for arg in specific_args]
            model_tag = ';'.join([f"{arg}={val}" for arg, val in zip(specific_args, specific_args_val)])
            if model_tag not in all_model_result:
                all_model_result[model_tag] = {}
                all_model_result[model_tag]['args'] = specific_args_val
                all_model_result[model_tag]['test_metric'] = {}

            # find best ep
            val_metric = log['valid_metric']
            best_ep = np.argmax([x[best_by] for x in val_metric])
            seed = args['seed']
            best_test_result = log['test_metric'][best_ep]
            all_model_result[model_tag]['test_metric'][seed] = best_test_result
    
    if len(all_task_result) == 0:
        print("No OOD datasets found with the specified marker.")
        exit(0)
    # convert to DataFrame
    all_df = {
        'MSE': {},
        'MAE': {},
        'R2': {},
        'RMSE': {}
    }
    for task, all_model_result in all_task_result.items():
        keep_cols = []
        for i, col in enumerate(specific_args):
            all_values = set()
            for k, v in all_model_result.items():
                all_values.add(v['args'][i])
            if len(all_values) > 1:
                keep_cols.append(col)
        
        record = {
            'MSE': [],
            'MAE': [],
            'R2': [],
            'RMSE': []
        }
        for old_tag, val in all_model_result.items():
            args_val = val['args']
            # item = {'model_tag': ';'.join([f"{c}={args}" )}
            mol_tag_dict = {col: args_val[specific_args.index(col)] for col in keep_cols}
            new_tag = ';'.join([f"{k}={v}" for k, v in mol_tag_dict.items()])
            item = {'model_tag': new_tag}

            item.update(
                dict(zip(specific_args, args_val))  # Add specific args
            )

            for m in record.keys():
                item_cp = item.copy()
                this_exp_all_metric = []
                for dataset, metric in val['test_metric'].items():
                    if m == 'RMSE':
                        this_exp_metrix = np.sqrt(metric['MSE'])
                    else:
                        this_exp_metrix = metric[m]
                    this_exp_all_metric.append(this_exp_metrix)
                    item_cp.update({dataset : this_exp_metrix})
                item_cp['mean'] = np.mean(this_exp_all_metric)
                item_cp['std'] = np.std(this_exp_all_metric)
                record[m].append(item_cp)
        for m in all_df.keys():
            df = pd.DataFrame(record[m])
            all_df[m][task] = df
    
    for m, df_dict in all_df.items():
        # save as excel, task name as sheet name
        with pd.ExcelWriter(os.path.join(input_file, f'collect_ood_{m}_best_by_{best_by}.xlsx')) as writer:
            for task, df in df_dict.items():
                df.to_excel(writer, sheet_name=task, index=False)