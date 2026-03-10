#!/bin/bash
set -e

batch_size=512
device=-1
result_dir=""
checkpoint_dir=""
data_path=""

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script runs inference and evaluation on USPTO-Yield using four fixed
ablation checkpoints. It reports MAE, RMSE, and R2 for each checkpoint
and prints a summary table (no mean/std; ablation study). Missing checkpoints
are skipped.

Options:
  --result_dir PATH       Directory to store results (required)
  --checkpoint_dir PATH   Path to directory containing .pth checkpoint files (required)
  --data_path PATH        Path to USPTO-Yield JSONL file (required)
  --batch_size INT        Batch size for inference (default: 512)
  --device INT            Device ID (-1 for CPU) (default: -1)
  --help                  Show this help message

Examples:
  $0 --result_dir ./results/uspto_yield \
     --checkpoint_dir ./checkpoints/uspto_yield \
     --data_path ./data/uspto-yield/alldata_nodup.jsonl
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --result_dir)
            result_dir="$2"
            shift 2
            ;;
        --checkpoint_dir)
            checkpoint_dir="$2"
            shift 2
            ;;
        --data_path)
            data_path="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$result_dir" ]] || [[ -z "$checkpoint_dir" ]] || [[ -z "$data_path" ]]; then
    echo "Error: --result_dir, --checkpoint_dir, and --data_path are required."
    usage
fi

if [[ ! -f "$data_path" ]]; then
    echo "Error: data file not found: $data_path"
    exit 1
fi
if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Error: checkpoint directory not found: $checkpoint_dir"
    exit 1
fi

mkdir -p "$result_dir"

script_path=$(realpath "$0")
script_dir=$(dirname "$(dirname "$script_path")")

declare -A ckpt_amount_map ckpt_temp_map
ckpt_amount_map["wo_amount_wo_temperature.pth"]=-1
ckpt_temp_map["wo_amount_wo_temperature.pth"]=-1
ckpt_amount_map["with_amount_wo_temperature.pth"]=50
ckpt_temp_map["with_amount_wo_temperature.pth"]=-1
ckpt_amount_map["wo_amount_with_temperature.pth"]=-1
ckpt_temp_map["wo_amount_with_temperature.pth"]=50
ckpt_amount_map["with_amount_with_temperature.pth"]=50
ckpt_temp_map["with_amount_with_temperature.pth"]=50

checkpoint_files=(
    "wo_amount_wo_temperature.pth"
    "with_amount_wo_temperature.pth"
    "wo_amount_with_temperature.pth"
    "with_amount_with_temperature.pth"
)

declare -A mae_map rmse_map r2_map
successful=()
failed=()

for filename in "${checkpoint_files[@]}"; do
    checkpoint_file="$checkpoint_dir/$filename"
    if [[ ! -f "$checkpoint_file" ]]; then
        echo "Skipping missing checkpoint: $filename"
        continue
    fi

    echo "Processing checkpoint $filename ..."
    base="${filename%.pth}"
    output_file="$result_dir/${base}.json"
    log_file="$result_dir/${base}.log"

    cmd=(python "$script_dir/predict_uspto_yield.py"
        --data_path "$data_path"
        --checkpoint "$checkpoint_file"
        --output_path "$output_file"
        --bs "$batch_size"
        --device "$device"
        --amount_class "${ckpt_amount_map[$filename]}"
        --temperature_class "${ckpt_temp_map[$filename]}"
    )

    set +e
    "${cmd[@]}" > "$log_file" 2>&1
    exit_code=$?
    set -e

    if [[ $exit_code -ne 0 ]]; then
        echo "Error: Failed to run inference for checkpoint $filename"
        failed+=("$filename")
        continue
    fi

    mae_line=$(grep -E '^MAE:' "$log_file" | tail -n1)
    rmse_line=$(grep -E '^RMSE:' "$log_file" | tail -n1)
    r2_line=$(grep -E '^R2:' "$log_file" | tail -n1)

    if [[ -z "$mae_line" || -z "$rmse_line" || -z "$r2_line" ]]; then
        echo "Warning: Could not parse metrics from log for checkpoint $filename"
        failed+=("$filename")
        continue
    fi

    mae=$(echo "$mae_line" | awk '{print $2}')
    rmse=$(echo "$rmse_line" | awk '{print $2}')
    r2=$(echo "$r2_line" | awk '{print $2}')

    mae_map["$filename"]=$mae
    rmse_map["$filename"]=$rmse
    r2_map["$filename"]=$r2
    successful+=("$filename")
    echo "Success for $filename: MAE=$mae, RMSE=$rmse, R2=$r2"
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Errors occur when evaluate using the following ckpts: ${failed[*]}"
fi

if [[ ${#successful[@]} -eq 0 ]]; then
    echo "No successful runs."
    exit 0
fi

max_name_len=6
for name in "${successful[@]}"; do
    len=${#name}
    if (( len+2 > max_name_len )); then
        max_name_len=$((len+2))
    fi
done

declare -A mae_str rmse_str r2_str
all_mae_strs=()
all_rmse_strs=()
all_r2_strs=()
for name in "${successful[@]}"; do
    mae_str["$name"]=$(printf "%.4f" "${mae_map[$name]}")
    rmse_str["$name"]=$(printf "%.4f" "${rmse_map[$name]}")
    r2_str["$name"]=$(printf "%.4f" "${r2_map[$name]}")
    all_mae_strs+=("${mae_str[$name]}")
    all_rmse_strs+=("${rmse_str[$name]}")
    all_r2_strs+=("${r2_str[$name]}")
done

max_mae_len=3
max_rmse_len=4
max_r2_len=2
for s in "${all_mae_strs[@]}"; do
    len=${#s}
    if (( len > max_mae_len )); then max_mae_len=$len; fi
done
for s in "${all_rmse_strs[@]}"; do
    len=${#s}
    if (( len > max_rmse_len )); then max_rmse_len=$len; fi
done
for s in "${all_r2_strs[@]}"; do
    len=${#s}
    if (( len > max_r2_len )); then max_r2_len=$len; fi
done

col1_width=$max_name_len
col2_width=$((max_mae_len + 2))
col3_width=$((max_rmse_len + 2))
col4_width=$((max_r2_len + 2))

echo ""
echo "Results:"
printf "+-%s-+-%s-+-%s-+-%s-+\n" \
    "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"
printf "| %-*s |" "$col1_width" ""
printf " %*s |" "$col2_width" "MAE"
printf " %*s |" "$col3_width" "RMSE"
printf " %*s |\n" "$col4_width" "R2"
printf "+-%s-+-%s-+-%s-+-%s-+\n" \
    "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"

for name in "${successful[@]}"; do
    printf "| %*s |" "$col1_width" "$name"
    printf " %*s |" "$col2_width" "${mae_str[$name]}"
    printf " %*s |" "$col3_width" "${rmse_str[$name]}"
    printf " %*s |\n" "$col4_width" "${r2_str[$name]}"
done

printf "+-%s-+-%s-+-%s-+-%s-+\n" \
    "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
    "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"

if ls "$result_dir"/*.log 1> /dev/null 2>&1; then
    rm "$result_dir"/*.log
    echo "Cleaned up log files in $result_dir"
fi

echo "Done."
