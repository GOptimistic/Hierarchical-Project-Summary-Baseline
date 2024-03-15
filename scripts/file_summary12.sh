#!/bin/bash

# 设置作业的名称、输出文件等
#SBATCH --job-name="file12"
#SBATCH --output="./sbatch/fileSummary/%j.out"
#SBATCH --error="./sbatch/fileSummary/%j.err"
#SBATCH --gres=gpu:3

conda activate base
python get_file_summary_import.py --start_part_index 12 --node_num 3