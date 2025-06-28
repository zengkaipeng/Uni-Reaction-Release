import argparse
import os
import json
from utils import validate, COL_SHORT, SHORT2COL
import shutil

BASE_OUTPUT_DIR = 'log/cmp'
parser = argparse.ArgumentParser(description=' collect all logs, where the args meets the given filter')
parser.add_argument('--input', '-i', type=str, required=True, help='Input log dir')
parser.add_argument('--output', '-o', type=str, default='', help='Output log dir')
parser.add_argument('--filter', '-f', type=str, required=True, help='Filter for log args, split by space')
args = parser.parse_args()

input_dir = args.input
output_dir = args.output
    
filter_args = args.filter.split('&')
filter_args = {item.split('=')[0].strip() : item.split('=')[1].strip() for item in filter_args if item.strip()}  # Remove empty strings
if output_dir == '':
    output_dir = os.path.join(BASE_OUTPUT_DIR, args.filter.replace(' ', '_').replace('/', '_'))

def satisfy_filter(args, filter_args):
    """
    Check if the log args meet the filter criteria.
    """
    for arg in filter_args:
        org_arg = arg
        if org_arg not in args:
            org_arg = SHORT2COL.get(arg, arg)
        if org_arg not in args or str(args[org_arg]) != filter_args[arg]:
            return False
    return True

for root, dirs, files in os.walk(input_dir):
    if len(dirs) == 0:
        log = validate(root)
        if log is None:
            continue

        # Check if the log args meet the filter
        if satisfy_filter(log['args'], filter_args):
            # Copy the log.json to the output directory
            output_path = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_path, exist_ok=True)
            log_json_path = os.path.join(root, 'log.json')
            shutil.copy(log_json_path, output_path)

print(f"Filtered logs have been collected to {output_dir}.")
        