import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


MAX_FILE_LENGTH = 10
# content_length = []
# summary_length = []


def print_data_status(data):
    mean_value = np.mean(data)
    median_value = np.median(data)
    counts = np.bincount(data)
    sum_counts = np.sum(counts)
    first_mode = np.argmax(counts)
    first_mode_num = counts[first_mode]
    counts[first_mode] = 0  # 将众数的计数置为0
    second_mode = np.argmax(counts)
    second_mode_num = counts[second_mode]
    counts[second_mode] = 0  # 将第二众数的计数置为0
    third_mode = np.argmax(counts)
    third_mode_num = counts[third_mode]
    q8 = np.percentile(data, 80)
    q9 = np.percentile(data, 90)

    print("平均数:", mean_value)
    print("中位数:", median_value)
    print("众数:", first_mode, " 频数为:", first_mode_num, " 比例为:", first_mode_num / sum_counts * 1.0)
    print("第二众数:", second_mode, " 频数为:", second_mode_num, " 比例为:", second_mode_num / sum_counts * 1.0)
    print("第三众数:", third_mode, " 频数为:", third_mode_num, " 比例为:", third_mode_num / sum_counts * 1.0)
    print("最大值:", len(counts) - 1)
    print("80%分位数:", q8)
    print("90%分位数:", q9)

def handle_file_summaries(repo_name, file_summaries):
    file_summaries_object = json.loads(file_summaries)
    file_keys = file_summaries_object.keys()
    if len(file_keys) > MAX_FILE_LENGTH:
        file_keys = list(file_keys)[:MAX_FILE_LENGTH]
    result = 'The open-source project {} has {} most important files and every file has a summary. Please give me an introduction about the project based on the file summaries below. \n'.format(repo_name, len(file_keys))
    for file in file_keys:
        file_summary = file_summaries_object[file]
        # 加上package和file的信息
        result += 'file {} summary {}\n'.format(file, file_summary)
    return result


def write_to_json(train_df, json_output_path):
    train_json = []
    for index, row in train_df.iterrows():
        content = handle_file_summaries(row['repo_name'], row['file_summaries'])
        summary = row['repo_summary']
        json_object = {
            "content": content,
            "summary": summary
        }
        train_json.append(json.dumps(json_object))

    with open(json_output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_json))


csv_path = '/Users/guanzheng/cls_work/graduation_model/Hierarchical-Project-Summary-Baseline/src/file_summary_data/analyze_import_data/file_summary_all_chatglm.csv'
df = pd.read_csv(csv_path, header=0)
# 划分数据集
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_df))

valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=56)
print(len(valid_df))
print(len(test_df))
write_to_json(train_df, '/src/file_summary_data/analyze_import_data/chatglm/chatglm_train.json')
write_to_json(valid_df, '/src/file_summary_data/analyze_import_data/chatglm/chatglm_valid.json')
write_to_json(test_df, '/src/file_summary_data/analyze_import_data/chatglm/chatglm_test.json')


# print('###### Content level')
# print_data_status(content_length)
# print('###### Summary level')
# print_data_status(summary_length)