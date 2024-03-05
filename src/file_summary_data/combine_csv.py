import pandas as pd
import glob
from sklearn.model_selection import train_test_split
# 设置包含CSV文件的目录路径
csv_files_path = './file_data_java_test/*.csv'  # 请替换为你的CSV文件所在目录

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

# 划分数据集
train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
print(len(train_df))

valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=56)
print(len(valid_df))
print(len(test_df))

train_df.to_csv('./mini_train.csv', index=False)
valid_df.to_csv('./mini_valid.csv', index=False)
test_df.to_csv('./mini_test.csv', index=False)
