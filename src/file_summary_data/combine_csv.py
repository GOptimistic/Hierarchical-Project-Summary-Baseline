import pandas as pd
import glob
import json
from sklearn.model_selection import train_test_split
import re
import inflect
import nltk.stem.porter as pt


def to_digit(digit):
    i = inflect.engine()
    if digit.isdigit():
        output = i.number_to_words(digit)
    else:
        output = digit
    return output


# 分词，小写，获得词源
def process_token(str):
    # 把&转化为有意义的and
    str = str.replace("&", "and")
    # 去掉非数字字母空格符号，比如标点
    str = re.sub(r'[^\w\s]', ' ', str)
    # 处理驼峰命名的词，切分成多个单词
    str = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str)
    tokens = str.split()
    # 将数字转化为英文单词
    tokens = [to_digit(x) for x in tokens]
    # 转化为小写
    tokens = [x.lower() for x in tokens]
    # 词干提取
    # 波特词干提取器
    # stemmer = pt.PorterStemmer()
    # tokens = [stemmer.stem(x) for x in tokens]

    return ' '.join(tokens)


# 处理文件摘要
def handle_file_summaries(repo_name, file_summaries):
    repo_name = repo_name.split('/')[1]
    file_summaries_object = json.loads(file_summaries)

    file_summaries_result = 'project {} '.format(repo_name)
    for package in file_summaries_object.keys():
        package_info = file_summaries_object[package]
        file_summaries_result += 'package {} '.format(package)
        for file in package_info.keys():
            file_summary = package_info[file]
            # 加上package和file的信息
            file_summaries_result += 'file {} summary {} '.format(file, file_summary)
    return process_token(file_summaries_result)

# 处理summary
def handle_repo_summary(repo_summary):
    return process_token(repo_summary)


def handle_csv():
    # 设置包含CSV文件的目录路径
    csv_files_path = './file_data_java_test/*.csv'  # 请替换为你的CSV文件所在目录

    # 使用glob找到所有的csv文件
    csv_files = glob.glob(csv_files_path)

    # 创建一个空列表来存储每个文件的DataFrame
    data_frames = []

    # 逐个读取CSV文件
    for filename in csv_files:
        df = pd.read_csv(filename, header=0)
        # print(df)
        for index, row in df.iterrows():
            row['file_summaries'] = handle_file_summaries(row['repo_name'], row['file_summaries'])
            row['repo_summary'] = handle_repo_summary(row['repo_summary'])
        data_frames.append(df)

    # 合并所有DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    # combined_df.to_csv('./all.csv', index=False)
    # print(combined_df)
    # 划分数据集
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    print(len(train_df))

    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=56)
    print(len(valid_df))
    print(len(test_df))

    train_df.to_csv('./mini_train_one_level.csv', index=False)
    valid_df.to_csv('./mini_valid_one_level.csv', index=False)
    test_df.to_csv('./mini_test_one_level.csv', index=False)

if __name__ == '__main__':
    # file_summary = '{"ch.qos.logback.core.sift": {"SiftingJoranConfiguratorBase.java": "This Java code snippet is from the logback-android project and is used to configure and manage appenders within a logback configuration file. It includes methods for adding implicit rules, instance rules, checking for only one appender within a <sift> element, and configuring the logback system.", "DefaultDiscriminator.java": "The DefaultDiscriminator.java file contains two methods: getDiscriminatingValue and getKey, both of which return the constant value DEFAULT. These methods are used to provide default values when processing log events in the logback-android project.", "AbstractAppenderFactoryUsingJoran.java": "This Java code snippet defines a factory for creating log appenders in the logback-android project. It includes functions to remove a specified element from a list of events, retrieve the list of events, and build an appender using a configuration.", "SiftingAppenderBase.java": "This Java code snippet is a base class for log appenders, providing methods for setting and getting configuration parameters, starting and stopping the log appender, and appending events to the log. It also includes methods for getting the timestamp of an event and determining if an event marks the end of life.", "AbstractDiscriminator.java": "The AbstractDiscriminator.java file provides utility functions for starting and stopping a discriminator and checking if it is started."}, "ch.qos.logback.core.hook": {"DefaultShutdownHook.java": "This code snippet defines a DefaultShutdownHook class that provides a delay before shutting down the system. It has methods to get, set, and run the delay, and it sleeps for the specified duration before stopping the system.", "ShutdownHookBase.java": "The code snippet in ShutdownHookBase.java is a shutdown hook function that stops the Logback context by calling the stop() method on the context object, which is an instance of ContextBase. The function also adds an info message to indicate that the context is being closed via a shutdown hook."}, "ch.qos.logback.core.helpers": {"NOPAppender.java": "The code snippet is a method in the NOPAppender class, which implements an appender that does not write any events to the log file. It takes an event object as input and does not perform any action on it.", "ThrowableToStringArray.java": "This Java code snippet provides a function called `convert()` that converts a `Throwable` object into a String array. It uses helper functions `extract()` and `formatFirstLine()` to format the first line of the stack trace and to find the number of common frames between the current stack trace and the parent stack trace, respectively. The `extract()` function then adds the formatted first line and common frames to a `String` list, which is returned as a String array.", "Transform.java": "This Java code snippet provides functions for handling XML tags in strings, including escaping characters that need to be replaced with their HTML entities, such as ampersand, less than, greater than, and double quotes. The functions include `escapeTags`, `escapeTags` (for a StringBuffer), `appendEscapingCDATA`, and `appendEscapingCDATA`.", "CyclicBuffer.java": "This Java code snippet implements a circular buffer, which can grow or shrink dynamically to accommodate incoming elements. It provides methods to add, get, and get the maximum size of elements, and it can also convert the buffer to a list of elements."}}'
    # repo_name = 'tony19/logback-android'
    # print(handle_file_summaries(repo_name, file_summary))
    handle_csv()
