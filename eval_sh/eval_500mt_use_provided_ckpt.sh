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

This script performs inference and evaluation on the USPTO-500MT dataset.

Options:
  --result_dir PATH       Directory to store results (required)
  --mode {regenerate,use_current}
                          Operation mode: 'regenerate' runs inference, 
                          'use_current' only evaluates existing results.
                          (default: use_current)
  --checkpoint_dir PATH   Path to checkpoint directory containing model.pth and token.pkl (required if mode=regenerate)
  --data_path PATH        Path to test dataset file (required if mode=regenerate)
  --beam_size INT         Beam size for inference (default: 10)
  --batch_size INT        Batch size for inference (default: 128)
  --device INT            Device ID (-1 for CPU) (default: -1)
  --help                  Show this help message

Examples:
  # Use existing results
  $0 --result_dir ./results

  # Regenerate results and evaluate
  $0 --result_dir ./results --mode regenerate --checkpoint_dir ./checkpoint --data_path ./data/test.json --beam_size 5
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
    # 检查必要的模型文件是否存在（可选但推荐）
    if [[ ! -f "$checkpoint_dir/model.pth" ]]; then
        echo "Error: $checkpoint_dir/model.pth not found."
        exit 1
    fi
    if [[ ! -f "$checkpoint_dir/token.pkl" ]]; then
        echo "Error: $checkpoint_dir/token.pkl not found."
        exit 1
    fi
else  # use_current 模式
    if [[ ! -d "$result_dir" ]]; then
        echo "Error: result_dir '$result_dir' does not exist. Cannot use existing results."
        usage
    fi
fi

# 获取脚本的绝对路径并确定 script_dir (脚本所在目录的父目录)
script_path=$(realpath "$0")
script_dir=$(dirname "$(dirname "$script_path")")  # 父目录的父目录，即项目根目录

# 如果是 regenerate 模式，运行推理
if [[ "$mode" == "regenerate" ]]; then
    echo "Running inference with checkpoint $checkpoint_dir on data $data_path"
    python "$script_dir/inference_uspto_500mt.py" \
        --dim 256 \
        --heads 8 \
        --n_layer 6 \
        --local 4 \
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
python "$script_dir/eval_500mt.py" \
    --file "$result_dir" \
    --beam "$beam_size"

echo "Done."