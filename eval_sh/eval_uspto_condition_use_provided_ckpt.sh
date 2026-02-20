#!/bin/bash
set -e  # 遇到错误立即退出

# 默认值
mode="use_current"
beam_size=10
batch_size=128
device=-1
result_dir=""
checkpoint_dir=""
data_path=""

# 用法函数
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script performs inference and evaluation on the USPTO-Condition dataset.

Options:
  --result_dir PATH       Path to the output result file (e.g., ./results/full.json) (required)
  --mode {regenerate,use_current}
                          Operation mode: 'regenerate' runs inference,
                          'use_current' only evaluates existing results.
                          (default: use_current)
  --checkpoint_dir PATH   Path to checkpoint directory, containing model.pth and token.pkl (required if mode=regenerate)
  --data_path PATH        Path to the csv file of dataset (required if mode=regenerate)
  --beam_size INT         Beam size for inference (default: 10)
  --batch_size INT        Batch size for inference (default: 128)
  --device INT            Device ID (-1 for CPU) (default: -1)
  --help                  Show this help message

Examples:
  # Use existing results
  $0 --result_dir ./results/full.json

  # Regenerate results and evaluate
  $0 --result_dir ./results/full.json --mode regenerate --checkpoint_dir ./checkpoint --data_path ./data/test.csv --beam_size 5
EOF
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --result_dir)
            result_dir="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            if [[ "$mode" != "regenerate" && "$mode" != "use_current" ]]; then
                echo "Error: mode must be 'regenerate' or 'use_current'"
                usage
            fi
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
        --beam_size)
            beam_size="$2"
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

# 检查必需参数
if [[ -z "$result_dir" ]]; then
    echo "Error: --result_dir is required."
    usage
fi

# 根据模式进行参数验证
if [[ "$mode" == "regenerate" ]]; then
    if [[ -z "$checkpoint_dir" ]]; then
        echo "Error: --checkpoint_dir is required when mode=regenerate."
        usage
    fi
    if [[ -z "$data_path" ]]; then
        echo "Error: --data_path is required when mode=regenerate."
        usage
    fi
    # 确保输出文件的父目录存在
    output_dir=$(dirname "$result_dir")
    if [[ ! -d "$output_dir" ]]; then
        mkdir -p "$output_dir"
    fi
else  # use_current 模式
    if [[ ! -f "$result_dir" ]]; then
        echo "Error: result file '$result_dir' does not exist. Cannot use existing results."
        exit 1
    fi
fi

# 获取脚本的绝对路径并确定 script_dir (脚本所在目录的父目录)
script_path=$(realpath "$0")
script_dir=$(dirname "$(dirname "$script_path")")  # 父目录的父目录，通常为项目根目录

# 如果是 regenerate 模式，运行推理
if [[ "$mode" == "regenerate" ]]; then
    # 可选：检查必需的模型文件是否存在
    if [[ ! -f "$checkpoint_dir/model.pth" ]]; then
        echo "Warning: $checkpoint_dir/model.pth not found. Inference may fail."
    fi
    if [[ ! -f "$checkpoint_dir/token.pkl" ]]; then
        echo "Warning: $checkpoint_dir/token.pkl not found. Inference may fail."
    fi

    echo "Running inference with checkpoint $checkpoint_dir on data $data_path"
    python "$script_dir/inference_condition.py" \
        --dim 384 \
        --heads 6 \
        --n_layer 6 \
        --local 3 \
        --device "$device" \
        --check "$checkpoint_dir/model.pth" \
        --token_ckpt "$checkpoint_dir/token.pkl" \
        --save 10000 \
        --output_file "$result_dir" \
        --beam "$beam_size" \
        --batch_size "$batch_size" \
        --num_w 4 \
        --data_path "$data_path"
fi

# 运行评估
echo "Evaluating results in $result_dir with beam size $beam_size"
python "$script_dir/evaluate_condition.py" \
    --file "$result_dir" \
    --beam "$beam_size"

python "$script_dir/evaluate_pred_split.py" \
    --file "$result_dir" \
    --beam "$beam_size"

echo "Done."