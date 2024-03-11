import pandas as pd
import glob
import json
from sklearn.model_selection import train_test_split
import re
import inflect
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import numpy as np

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
flat_input_length = []
repo_summary_length = []

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res


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
    # 处理下划线
    str = str.replace('_', ' ')
    # 处理驼峰命名的词，切分成多个单词
    str = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str)
    tokens = str.split()
    # 将数字转化为英文单词
    tokens = [to_digit(x) for x in tokens]
    # 转化为小写
    tokens = [x.lower() for x in tokens]
    # # 词性还原
    # sentence = ' '.join(tokens)
    # tokens = lemmatize_sentence(sentence)
    # 词干提取
    tokens = [porter_stemmer.stem(x) for x in tokens]

    return ' '.join(tokens)


# 处理flat_input
def handle_flat_input(flat_input):
    global flat_input_length
    res = process_token(flat_input)
    flat_input_length.append(len(res))
    return res


# 处理summary
def handle_repo_summary(repo_summary):
    global repo_summary_length
    res = process_token(repo_summary)
    repo_summary_length.append(len(res))
    return res


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

def handle_csv():
    # 设置包含CSV文件的目录路径
    csv_file_path = '../src/flat_input_data/mini_all_flat.csv'  # 请替换为你的CSV文件所在目录

    df = pd.read_csv(csv_file_path, header=0)
    for index, row in df.iterrows():
        row['repo_summary'] = handle_repo_summary(row['repo_summary'])
        row['flat_input'] = handle_flat_input(row['flat_input'])

    df.to_csv('../src/flat_input_data/mini_all_flat_formatted.csv', index=False)
    print(len(df))
    # 划分数据集
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_df))

    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=56)
    print(len(valid_df))
    print(len(test_df))

    train_df.to_csv('../src/flat_input_data/mini_train_flat.csv', index=False)
    valid_df.to_csv('../src/flat_input_data/mini_valid_flat.csv', index=False)
    test_df.to_csv('../src/flat_input_data/mini_test_flat.csv', index=False)

if __name__ == '__main__':
    # file_summary = '{"ch.qos.logback.core.sift": {"SiftingJoranConfiguratorBase.java": "This Java code snippet is from the logback-android project and is used to configure and manage appenders within a logback configuration file. It includes methods for adding implicit rules, instance rules, checking for only one appender within a <sift> element, and configuring the logback system.", "DefaultDiscriminator.java": "The DefaultDiscriminator.java file contains two methods: getDiscriminatingValue and getKey, both of which return the constant value DEFAULT. These methods are used to provide default values when processing log events in the logback-android project.", "AbstractAppenderFactoryUsingJoran.java": "This Java code snippet defines a factory for creating log appenders in the logback-android project. It includes functions to remove a specified element from a list of events, retrieve the list of events, and build an appender using a configuration.", "SiftingAppenderBase.java": "This Java code snippet is a base class for log appenders, providing methods for setting and getting configuration parameters, starting and stopping the log appender, and appending events to the log. It also includes methods for getting the timestamp of an event and determining if an event marks the end of life.", "AbstractDiscriminator.java": "The AbstractDiscriminator.java file provides utility functions for starting and stopping a discriminator and checking if it is started."}, "ch.qos.logback.core.hook": {"DefaultShutdownHook.java": "This code snippet defines a DefaultShutdownHook class that provides a delay before shutting down the system. It has methods to get, set, and run the delay, and it sleeps for the specified duration before stopping the system.", "ShutdownHookBase.java": "The code snippet in ShutdownHookBase.java is a shutdown hook function that stops the Logback context by calling the stop() method on the context object, which is an instance of ContextBase. The function also adds an info message to indicate that the context is being closed via a shutdown hook."}, "ch.qos.logback.core.helpers": {"NOPAppender.java": "The code snippet is a method in the NOPAppender class, which implements an appender that does not write any events to the log file. It takes an event object as input and does not perform any action on it.", "ThrowableToStringArray.java": "This Java code snippet provides a function called `convert()` that converts a `Throwable` object into a String array. It uses helper functions `extract()` and `formatFirstLine()` to format the first line of the stack trace and to find the number of common frames between the current stack trace and the parent stack trace, respectively. The `extract()` function then adds the formatted first line and common frames to a `String` list, which is returned as a String array.", "Transform.java": "This Java code snippet provides functions for handling XML tags in strings, including escaping characters that need to be replaced with their HTML entities, such as ampersand, less than, greater than, and double quotes. The functions include `escapeTags`, `escapeTags` (for a StringBuffer), `appendEscapingCDATA`, and `appendEscapingCDATA`.", "CyclicBuffer.java": "This Java code snippet implements a circular buffer, which can grow or shrink dynamically to accommodate incoming elements. It provides methods to add, get, and get the maximum size of elements, and it can also convert the buffer to a list of elements."}}'
    # repo_name = 'tony19/logback-android'
    # print(handle_file_summaries(repo_name, file_summary))
    handle_csv()
    # 打印信息
    print('###### Flat input level')
    print_data_status(flat_input_length)
    print('###### Repo summary level')
    print_data_status(repo_summary_length)
