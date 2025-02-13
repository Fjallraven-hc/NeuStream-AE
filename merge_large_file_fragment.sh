#!/bin/bash

# 定义需要合并的文件夹
DIR="."  # 当前目录，可以根据需要修改为其他路径

# 查找所有以 _part_ 开头的文件
find "$DIR" -type f -name "*_part_*" | while read -r part_file; do
    # 获取原始文件名（去掉 _part_ 及后缀）
    original_file=$(echo "$part_file" | sed 's/_part_.*//')

    # 检查是否已经存在合并后的文件
    if [[ -f "$original_file" ]]; then
        echo "合并后的文件已存在: $original_file，跳过合并。"
        continue
    fi

    # 合并所有部分文件
    echo "正在合并文件: $original_file"
    cat "${original_file}_part_"* > "$original_file"

    # 检查合并是否成功
    if [[ $? -eq 0 ]]; then
        echo "合并成功: $original_file"
    else
        echo "合并失败: $original_file"
    fi
done