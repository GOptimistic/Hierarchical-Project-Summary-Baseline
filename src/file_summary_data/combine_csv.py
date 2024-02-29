import pandas as pd
import glob

# 设置包含CSV文件的目录路径
csv_files_path = './file_data/*.csv'  # 请替换为你的CSV文件所在目录

# 使用glob找到所有的csv文件
csv_files = glob.glob(csv_files_path)

# 创建一个空列表来存储每个文件的DataFrame
data_frames = []

# 逐个读取CSV文件
for filename in csv_files:
    df = pd.read_csv(filename, header=0)
    data_frames.append(df)

# 合并所有DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)

# 将合并后的DataFrame保存到新的CSV文件
combined_csv_path = './combined_csv.csv'  # 你可以更改这个文件名
combined_df.to_csv(combined_csv_path, index=False)

print(f'All files have been combined into {combined_csv_path}')
