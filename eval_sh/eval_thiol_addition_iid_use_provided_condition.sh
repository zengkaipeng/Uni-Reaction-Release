#!/bin/bash
set -e

# 默认值
batch_size=128
device=-1
use_pretrain=false
result_dir=""
checkpoint_dir=""
data_path=""

# 用法函数
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script performs inference and evaluation on multiple data splits of the chiral phosphoric acid-catalyzed thiol 
addition dataset using corresponding checkpoint files. Each checkpoint file should be named as model_<split>.pth,
and the data for that split should be in <data_path>/<split>/ (containing train.csv, val.csv, test.csv).

Options:
  --result_dir PATH       Directory to store results (required)
  --checkpoint_dir PATH   Path to directory containing model_*.pth checkpoint files (required)
  --data_path PATH        Path to parent directory containing split subfolders (required)
  --batch_size INT        Batch size for inference (default: 128)
  --device INT            Device ID (-1 for CPU) (default: -1)
  --use_pretrain          Use pretrained condition encoder (flag)
  --help                  Show this help message

Examples:
  $0 --result_dir ./results --checkpoint_dir ./checkpoints --data_path ./data/cv_folds
  $0 --result_dir ./results --checkpoint_dir ./checkpoints --data_path ./data --use_pretrain
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
        --use_pretrain)
            use_pretrain=true
            shift
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
if [[ -z "$result_dir" ]] || [[ -z "$checkpoint_dir" ]] || [[ -z "$data_path" ]]; then
    echo "Error: --result_dir, --checkpoint_dir, and --data_path are required."
    usage
fi

# 获取脚本的绝对路径并确定 script_dir (脚本所在目录的父目录)
script_path=$(realpath "$0")
script_dir=$(dirname "$(dirname "$script_path")")  # 脚本所在目录的父目录

# 创建结果目录（如果不存在）
mkdir -p "$result_dir"

# 确定 condition_config 路径
if $use_pretrain; then
    # 预训练模式：复制配置文件并替换 masking.pth 路径
    pretrain_config_src="$script_dir/condition_config/dm/config_dm_pretrain.json"
    if [[ ! -f "$pretrain_config_src" ]]; then
        echo "Error: Pretrain config not found at $pretrain_config_src"
        exit 1
    fi
    temp_config="$result_dir/temp_condition_config.json"
    cp "$pretrain_config_src" "$temp_config"
    # 替换 "condition_config/masking.pth" 为 "script_dir/condition_config/masking.pth"
    masking_path="$script_dir/condition_config/masking.pth"
    original_pattern="condition_config/masking.pth"
    escaped_original=$(echo "$original_pattern" | sed 's/\//\\\//g')
    escaped_masking=$(echo "$masking_path" | sed 's/\//\\\//g')
    sed -i "s/$escaped_original/$escaped_masking/g" "$temp_config"
    condition_config_path="$temp_config"
else
    # 非预训练模式：直接使用无预训练的配置文件
    no_pretrain_config="$script_dir/condition_config/dm/config_dm_no_pretrain_gat.json"
    if [[ ! -f "$no_pretrain_config" ]]; then
        echo "Error: No-pretrain config not found at $no_pretrain_config"
        exit 1
    fi
    condition_config_path="$no_pretrain_config"
fi

echo "Using condition config: $condition_config_path"

# 查找 checkpoint_dir 下所有 model_*.pth 文件
checkpoint_files=()
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    # 只匹配以 model_ 开头、.pth 结尾的文件
    if [[ "$filename" == model_*.pth ]]; then
        checkpoint_files+=("$filename")
    fi
done < <(find "$checkpoint_dir" -maxdepth 1 -name "model_*.pth" -type f -print0)

if [[ ${#checkpoint_files[@]} -eq 0 ]]; then
    echo "Error: No model_*.pth files found in $checkpoint_dir"
    exit 1
fi

# 初始化成功和失败列表
successful=()
failed=()
declare -A mae_map rmse_map r2_map

# 遍历每个 checkpoint 文件
for filename in "${checkpoint_files[@]}"; do
    # 从文件名中提取 split 名称（去掉 model_ 前缀和 .pth 后缀）
    # 例如 model_fold1.pth -> fold1
    split="${filename#model_}"
    split="${split%.pth}"

    echo "Processing split $split using checkpoint $filename ..."

    # 检查 data_path 下是否存在同名子文件夹
    data_subdir="$data_path/$split"
    if [[ ! -d "$data_subdir" ]]; then
        echo "Warning: Data subdirectory $data_subdir not found for split $split, skipping."
        continue
    fi

    output_file="$result_dir/${split}.json"
    checkpoint_file="$checkpoint_dir/$filename"
    log_file="$result_dir/${split}.log"

    # 运行推理脚本
    set +e  # 暂时关闭 exit on error
    python "$script_dir/predict_dm.py" \
        --data_path "$data_subdir" \
        --bs "$batch_size" \
        --device "$device" \
        --condition_config "$condition_config_path" \
        --output "$output_file" \
        --checkpoint "$checkpoint_file" > "$log_file" 2>&1
    exit_code=$?
    set -e  # 重新启用 exit on error

    if [[ $exit_code -ne 0 ]]; then
        echo "Error: Failed to run inference for split $split"
        failed+=("$split")
        continue
    fi

    # 从日志中提取 MAE, RMSE, R2
    mae_line=$(grep -E '^MAE:' "$log_file" | tail -n1)
    rmse_line=$(grep -E '^RMSE:' "$log_file" | tail -n1)
    r2_line=$(grep -E '^R2:' "$log_file" | tail -n1)

    if [[ -z "$mae_line" || -z "$rmse_line" || -z "$r2_line" ]]; then
        echo "Warning: Could not parse metrics from log for split $split"
        failed+=("$split")
        continue
    fi

    mae=$(echo "$mae_line" | awk '{print $2}')
    rmse=$(echo "$rmse_line" | awk '{print $2}')
    r2=$(echo "$r2_line" | awk '{print $2}')

    # 存储结果，键使用 split 名称
    mae_map["$split"]=$mae
    rmse_map["$split"]=$rmse
    r2_map["$split"]=$r2
    successful+=("$split")
    echo "Success for split $split: MAE=$mae, RMSE=$rmse, R2=$r2"
done

# 总结
if [[ ${#successful[@]} -eq 0 && ${#failed[@]} -eq 0 ]]; then
    echo "No Data Found"
    exit 0
fi

# 如果有失败项
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "下面这些 split 运行时出错： ${failed[*]}"
fi

# 如果有成功项，计算统计并制表
if [[ ${#successful[@]} -gt 0 ]]; then
    # 收集所有成功项的值
    mae_values=()
    rmse_values=()
    r2_values=()
    for split in "${successful[@]}"; do
        mae_values+=("${mae_map[$split]}")
        rmse_values+=("${rmse_map[$split]}")
        r2_values+=("${r2_map[$split]}")
    done

    # 计算均值（使用 awk）
    compute_mean() {
        local values=("$@")
        printf '%s\n' "${values[@]}" | awk '{sum+=$1} END {printf "%.4f", sum/NR}'
    }
    # 计算标准差
    compute_std() {
        local values=("$@")
        local mean=$1
        shift
        printf '%s\n' "$@" | awk -v mean="$mean" '{sum+=($1-mean)^2} END {printf "%.4f", sqrt(sum/NR)}'
    }

    mae_mean=$(compute_mean "${mae_values[@]}")
    rmse_mean=$(compute_mean "${rmse_values[@]}")
    r2_mean=$(compute_mean "${r2_values[@]}")

    # 计算列宽
    # 第一列：max(6, 最长 split 名称长度+2)
    max_name_len=6
    for split in "${successful[@]}"; do
        len=${#split}
        if (( len+2 > max_name_len )); then
            max_name_len=$((len+2))
        fi
    done

    # 准备所有单元格的字符串（保留四位小数）
    declare -A mae_str rmse_str r2_str
    all_mae_strs=()
    all_rmse_strs=()
    all_r2_strs=()
    for split in "${successful[@]}"; do
        mae_str["$split"]=$(printf "%.4f" "${mae_map[$split]}")
        rmse_str["$split"]=$(printf "%.4f" "${rmse_map[$split]}")
        r2_str["$split"]=$(printf "%.4f" "${r2_map[$split]}")
        all_mae_strs+=("${mae_str[$split]}")
        all_rmse_strs+=("${rmse_str[$split]}")
        all_r2_strs+=("${r2_str[$split]}")
    done

    # 计算后面三列的宽度
    max_mae_len=3  # 至少 "MAE" 长度
    max_rmse_len=4 # "RMSE"
    max_r2_len=2   # "R2"
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
    # 加2为两侧空格
    col1_width=$max_name_len
    col2_width=$((max_mae_len + 2))
    col3_width=$((max_rmse_len + 2))
    col4_width=$((max_r2_len + 2))

    echo ""
    echo "Results:"
    # 上边框
    printf "+-%s-+-%s-+-%s-+-%s-+\n" \
        "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"
    # 表头
    printf "| %-*s |" "$col1_width" ""
    printf " %*s |" "$col2_width" "MAE"
    printf " %*s |" "$col3_width" "RMSE"
    printf " %*s |\n" "$col4_width" "R2"
    # 表头下边框
    printf "+-%s-+-%s-+-%s-+-%s-+\n" \
        "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"

    # 数据行
    for split in "${successful[@]}"; do
        printf "| %*s |" "$col1_width" "$split"
        printf " %*s |" "$col2_width" "${mae_str[$split]}"
        printf " %*s |" "$col3_width" "${rmse_str[$split]}"
        printf " %*s |\n" "$col4_width" "${r2_str[$split]}"
    done

    # 数据行和统计行之间的分割线
    printf "+-%s-+-%s-+-%s-+-%s-+\n" \
        "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"

    # 均值行
    printf "| %*s |" "$col1_width" "mean"
    printf " %*s |" "$col2_width" "$mae_mean"
    printf " %*s |" "$col3_width" "$rmse_mean"
    printf " %*s |\n" "$col4_width" "$r2_mean"

    # 如果有至少两个成功项，计算标准差
    if [[ ${#successful[@]} -gt 1 ]]; then
        mae_std=$(compute_std "${mae_mean}" "${mae_values[@]}")
        rmse_std=$(compute_std "${rmse_mean}" "${rmse_values[@]}")
        r2_std=$(compute_std "${r2_mean}" "${r2_values[@]}")
        printf "| %*s |" "$col1_width" "std"
        printf " %*s |" "$col2_width" "$mae_std"
        printf " %*s |" "$col3_width" "$rmse_std"
        printf " %*s |\n" "$col4_width" "$r2_std"
    fi

    # 下边框
    printf "+-%s-+-%s-+-%s-+-%s-+\n" \
        "$(printf '%*s' "$col1_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col2_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col3_width" '' | tr ' ' '-')" \
        "$(printf '%*s' "$col4_width" '' | tr ' ' '-')"
fi

# 清理临时文件
if $use_pretrain && [[ -f "$temp_config" ]]; then
    rm "$temp_config"
    echo "Cleaned up temporary config: $temp_config"
fi
# 删除所有 .log 文件
if ls "$result_dir"/*.log 1> /dev/null 2>&1; then
    rm "$result_dir"/*.log
    echo "Cleaned up log files in $result_dir"
fi

echo "Done."