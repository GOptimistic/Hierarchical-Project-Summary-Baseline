import pandas as pd
import glob
import json
from sklearn.model_selection import train_test_split
import numpy as np
import re

filter_pattern = r'archived|deprecated|unmaintained|unsupported'
file_length = []
flat_input_length = []
max_file_num = 30


#   处理文件信息，将方法签名去掉，只保留package名、degree和对应的路径
def handle_files_info(files_info):
    files_info_object = json.loads(files_info)

    for file in files_info_object.keys():
        file_info = files_info_object[file]
        if 'methods' in file_info:
            file_info.pop('methods')
        files_info_object[file] = file_info
    return json.dumps(files_info_object)


#   将输入拉平
def get_project_flat_input(repo_name, files_info):
    global file_length
    global flat_input_length
    repo_name = repo_name.split('/')[1]

    files_info_object = json.loads(files_info)
    files_info_items = files_info_object.items()
    flat_input = 'project {} '.format(repo_name)
    # files = files_info_object.keys()
    file_length.append(len(files_info_items))
    if len(files_info_items) > max_file_num:
        files_info_items = sorted(files_info_items, key=lambda x: x[1]["degree"], reverse=True)[:max_file_num]
    for file_name, file_info in files_info_items:
        methods = file_info['methods']
        flat_input += 'class {} method {} '.format(file_name, ' '.join(methods))

    flat_input_length.append(len(flat_input.split()))
    return flat_input


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


def write_flat_to_txt(src_path, df_src, tgt_path, df_tgt):
    with open(src_path, 'w') as f:
        f.write('\n'.join(df_src.astype(str)))

    with open(tgt_path, 'w') as f:
        f.write('\n'.join(df_tgt.astype(str)))


def handle_csv():
    # 设置包含CSV文件的目录路径
    csv_files_path = '/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/part_data/*.csv'  # 请替换为你的CSV文件所在目录

    # 使用glob找到所有的csv文件
    csv_files = glob.glob(csv_files_path)
    print(len(csv_files))
    # 创建一个空列表来存储每个文件的DataFrame
    data_frames = []
    flat_result_list = []
    # 逐个读取CSV文件
    for filename in csv_files:
        try:
            df = pd.read_csv(filename, header=0)
        except Exception as e:
            print(filename)
        # print(df)
        for index, row in df.iterrows():
            repo_name = row['repo_name']
            files_info = row['files_info']
            repo_summary = ' '.join(row['repo_summary'].split())
            if not re.match(r'\$', repo_summary) and not re.match(filter_pattern, repo_summary, flags=re.IGNORECASE):
                flat_result_list.append([repo_name, get_project_flat_input(repo_name, files_info), repo_summary])
            row['files_info'] = handle_files_info(files_info)
            row['repo_summary'] = repo_summary
        filtered_df = df[
            ~df['repo_summary'].str.contains('\$', regex=True) & ~df['repo_summary'].str.contains(filter_pattern,
                                                                                                  flags=re.IGNORECASE,
                                                                                                  regex=True)]
        data_frames.append(filtered_df)

    # 合并所有DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.to_csv('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_analyze_all.csv',
                       index=False)
    print(len(combined_df))
    # 划分数据集
    # train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    # print(len(train_df))
    #
    # valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=56)
    # print(len(valid_df))
    # print(len(test_df))
    #
    # train_df.to_csv('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_analyze_train.csv',
    #                 index=False)
    # valid_df.to_csv('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_analyze_valid.csv',
    #                 index=False)
    # test_df.to_csv('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_analyze_test.csv',
    #                index=False)

    # 处理flat_input
    # 转换为DataFrame
    flat_df = pd.DataFrame(flat_result_list, columns=['repo_name', 'flat_input', 'repo_summary'])
    flat_df.to_csv('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_flat_input.csv',
                   index=False)
    print(len(flat_df))
    # 划分数据集
    flat_train_df, flat_temp_df = train_test_split(flat_df, test_size=0.2, random_state=42)
    print(len(flat_train_df))

    flat_valid_df, flat_test_df = train_test_split(flat_temp_df, test_size=0.5, random_state=56)
    print(len(flat_valid_df))
    print(len(flat_test_df))

    #   写入文本文件
    write_flat_to_txt('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_train_src.txt',
                      flat_train_df['flat_input'],
                      '/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_train_tgt.txt',
                      flat_train_df['repo_summary'])
    write_flat_to_txt('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_valid_src.txt',
                      flat_valid_df['flat_input'],
                      '/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_valid_tgt.txt',
                      flat_valid_df['repo_summary'])
    write_flat_to_txt('/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_test_src.txt',
                      flat_test_df['flat_input'],
                      '/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/flat_test_tgt.txt',
                      flat_test_df['repo_summary'])


if __name__ == '__main__':
    # file_summary = '{"ch.qos.logback.core.sift": {"SiftingJoranConfiguratorBase.java": "This Java code snippet is from the logback-android project and is used to configure and manage appenders within a logback configuration file. It includes methods for adding implicit rules, instance rules, checking for only one appender within a <sift> element, and configuring the logback system.", "DefaultDiscriminator.java": "The DefaultDiscriminator.java file contains two methods: getDiscriminatingValue and getKey, both of which return the constant value DEFAULT. These methods are used to provide default values when processing log events in the logback-android project.", "AbstractAppenderFactoryUsingJoran.java": "This Java code snippet defines a factory for creating log appenders in the logback-android project. It includes functions to remove a specified element from a list of events, retrieve the list of events, and build an appender using a configuration.", "SiftingAppenderBase.java": "This Java code snippet is a base class for log appenders, providing methods for setting and getting configuration parameters, starting and stopping the log appender, and appending events to the log. It also includes methods for getting the timestamp of an event and determining if an event marks the end of life.", "AbstractDiscriminator.java": "The AbstractDiscriminator.java file provides utility functions for starting and stopping a discriminator and checking if it is started."}, "ch.qos.logback.core.hook": {"DefaultShutdownHook.java": "This code snippet defines a DefaultShutdownHook class that provides a delay before shutting down the system. It has methods to get, set, and run the delay, and it sleeps for the specified duration before stopping the system.", "ShutdownHookBase.java": "The code snippet in ShutdownHookBase.java is a shutdown hook function that stops the Logback context by calling the stop() method on the context object, which is an instance of ContextBase. The function also adds an info message to indicate that the context is being closed via a shutdown hook."}, "ch.qos.logback.core.helpers": {"NOPAppender.java": "The code snippet is a method in the NOPAppender class, which implements an appender that does not write any events to the log file. It takes an event object as input and does not perform any action on it.", "ThrowableToStringArray.java": "This Java code snippet provides a function called `convert()` that converts a `Throwable` object into a String array. It uses helper functions `extract()` and `formatFirstLine()` to format the first line of the stack trace and to find the number of common frames between the current stack trace and the parent stack trace, respectively. The `extract()` function then adds the formatted first line and common frames to a `String` list, which is returned as a String array.", "Transform.java": "This Java code snippet provides functions for handling XML tags in strings, including escaping characters that need to be replaced with their HTML entities, such as ampersand, less than, greater than, and double quotes. The functions include `escapeTags`, `escapeTags` (for a StringBuffer), `appendEscapingCDATA`, and `appendEscapingCDATA`.", "CyclicBuffer.java": "This Java code snippet implements a circular buffer, which can grow or shrink dynamically to accommodate incoming elements. It provides methods to add, get, and get the maximum size of elements, and it can also convert the buffer to a list of elements."}}'
    # repo_name = 'tony19/logback-android'
    # print(handle_file_summaries(repo_name, file_summary))
    handle_csv()
    # 打印信息
    print('###### File level')
    print_data_status(file_length)
    print('###### Flat input level')
    print_data_status(flat_input_length)
