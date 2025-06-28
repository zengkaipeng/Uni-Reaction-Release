import os
import json
import argparse
import shutil

def validate(log_json):
    if not os.path.exists(log_json):
        return False
    with open(log_json, 'r') as f:
        log = json.load(f)
    if 'args' not in log or 'valid_metric' not in log or 'test_metric' not in log or 'train_loss' not in log:
        return False
    
    if len(set([log['args']['epoch'], len(log['train_loss']), len(log['valid_metric']), len(log['test_metric'])])) != 1:
        return False

    return True



parser = argparse.ArgumentParser(description='Clear invalid logs')
parser.add_argument('--input', '-i', type=str, required=True, help='Input log dir')
args = parser.parse_args()
input_dir = args.input
to_be_remove = []
for root, dirs, files in os.walk(input_dir):
    if len(dirs) == 0:
        log_json = os.path.join(root, 'log.json')
        if not validate(log_json):
            to_be_remove.append(root)

if len(to_be_remove) == 0:
    print("No invalid logs found.")
else:
    os.makedirs('log_remove', exist_ok=True)
    for log_dir in to_be_remove:
        shutil.move(log_dir, 'log_remove/')
        print(f"Moved invalid log: {log_dir} to log_remove/")
    print(f"Invalid logs have been moved to log_remove/.")