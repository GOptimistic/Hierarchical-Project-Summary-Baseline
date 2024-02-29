import re
from random import sample
from transformers import AutoTokenizer, AutoModel
import time
import json
import argparse
from multiprocessing import Process
import csv
import sys
import os
import torch

csv.field_size_limit(sys.maxsize)


# 读取data_java_output_filtered_with_json.csv对每一个仓库的repo_tree_info.json进行解析，得到file级别摘要，最终存在csv文件中
# 每个用户最多占有16个gpu，每个节点最多4个gpu，申请4个节点，将csv中读数据分成16片，每个任务处理的片段数用参数指明

# TODO：对util这些类可以去除，减少噪声
filterout_regex = 'util|test|docs'
def get_args():
    parser = argparse.ArgumentParser(
        """Get file summary of each project""")

    parser.add_argument("--node_num", type=int, default=2)
    parser.add_argument("--split_data_num", type=int, default=32)
    parser.add_argument("--data_part_size", type=int, default=1000)
    parser.add_argument("--start_part_index", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/chatglm3-6b-128k")
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--csv_data_path", type=str, default="/home/LAB/guanz/gz_graduation/clone_github_repo_data/java/data_java_output_filtered_with_json.csv")
    parser.add_argument("--output_data_path", type=str, default="/home/LAB/guanz/gz_graduation/model_file_summary/src/file_summary_data/file_data")
    parser.add_argument("--repo_path", type=str, default="/home/LAB/guanz/gz_graduation/clone_github_repo_data/github_repo_data")
    parser.add_argument("--max_length_package", type=int, default=5)
    parser.add_argument("--max_length_file", type=int, default=5)

    args = parser.parse_args()
    return args


def get_single_project_file_summary(json_path, model, tokenizer, lang, max_length_package, max_length_file):
    sequence_max_length = model.config.seq_length
    with open(json_path, 'r', encoding='utf-8') as f:
        text = f.read()
    project = json.loads(text)
    packages = project["packages"]
    project_name = project["full_name"].split('/')[1]
    file_summaries = {}
    package_keys = packages.keys()
    if len(package_keys) > max_length_package:
        package_keys = sample(package_keys, max_length_package)
    for package in package_keys:
        if len(re.findall(filterout_regex, package, re.IGNORECASE)) != 0:
            continue
        package_info = {}
        files = packages[package]['files']
        files_keys = files.keys()
        if len(files_keys) > max_length_file:
            files_keys = sample(files_keys, max_length_file)
        for file in files_keys:
            if len(re.findall(filterout_regex, file, re.IGNORECASE)) != 0:
                continue
            methods = files[file]['methods']
            methods_text = ''
            for method in methods:
                # method = method.replace('\n', '').replace('\r', '').replace('\t', '')
                methods_text = methods_text + method + '\n'
            # question = '请帮我对如下{}代码做一句话的内容摘要，这些代码是将{}文件中的函数拼接而成的，来自开源代码项目{}，所在的package是{}，要求是根据这些函数代码对整个文件做简短的一句话内容摘要，不需要对每一个函数做介绍，限制在30个单词以内，语言为英文：\n'.format(
            #     lang,
            #     file,
            #     project_name,
            #     package
            # )
            question = 'Please help me write a one-sentence summary of the following {} code snippet. The code is assembled from functions within a file named {} sourced from an open-source project named {}, residing in the package {}. The summary should succinctly capture the overall functionality of the file in under 30 words, without describing each function individually.\n\n'.format(
                lang,
                file,
                project_name,
                package
            )
            question = question + methods_text
            if len(tokenizer.tokenize(question)) >= sequence_max_length:
                continue
            response, history = model.chat(tokenizer, question, history=[])
            package_info[file] = response
        file_summaries[package] = package_info
    return file_summaries

def run_single_process(config, data_list, node_index):
    start_time = time.time()
    model_path = config.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
    device = "cuda:{}".format(node_index)
    print(device)
    model.to(device)
    model = model.eval()
    with open(config.output_data_path + os.sep + 'data_{}_file_summary_{}_{}.csv'.format(config.lang, config.start_part_index, node_index), 'w', newline='') as out_file:
        csv_witer = csv.writer(out_file)
        # head of csv
        csv_witer.writerow(['repo_name', 'file_summaries', 'repo_summary'])

        for i in range(len(data_list)):
            data = data_list[i]
            full_name = data[0]
            summary = data[1]
            json_path = config.repo_path + os.sep + full_name.replace('/', '_') + '/repo_tree_info.json'
            try:
                file_summaries = get_single_project_file_summary(json_path, model, tokenizer, config.lang, config.max_length_package, config.max_length_file)
                if len(file_summaries.keys()) == 0:
                    continue
                csv_witer.writerow([full_name, json.dumps(file_summaries), summary])
            except RuntimeError as exception:
                if "CUDA out of memory" in str(exception):
                    print("ERROR: out of memory. part {} node {} index {}".format(config.start_part_index, node_index, i))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise exception

            print('part {} node {} index {}'.format(config.start_part_index, node_index, i))

    end_time = time.time()
    print('###### Process {} has done. Use {}s'.format(node_index, end_time - start_time))

def run_multi_process(config):
    repos_and_summaries = []
    index = 0
    with open(config.csv_data_path, 'r') as in_file:
        csv_reader = csv.reader(in_file)
        for row in csv_reader:
            if index == 0:
                index = index + 1
                continue
            full_name = row[1]
            summary = row[9].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            repos_and_summaries.append((full_name, summary))
            # print((full_name, summary))
            index = index + 1

    print(len(repos_and_summaries))
    processes = []
    for i in range(config.node_num):
        start_index = (config.start_part_index + i) * config.data_part_size
        end_index = (config.start_part_index + i + 1) * config.data_part_size
        print('process {} start {} end {}'.format(i, start_index, end_index))
        if end_index > len(repos_and_summaries):
            end_index = len(repos_and_summaries)
        processes.append(Process(target=run_single_process, args=(config, repos_and_summaries[start_index:end_index], i,)))
    [p.start() for p in processes]  # 开启进程
    [p.join() for p in processes]  # 等待进程依次结束

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained('../pretrained/chatglm3-6b-128k', trust_remote_code=True)
    # print(tokenizer.vocab_size)
    # question = 'Please help me write a one-sentence summary of the following {} code snippet.'
    # print(tokenizer.tokenize(question))
    # model = AutoModel.from_pretrained('../pretrained/chatglm3-6b-128k', trust_remote_code=True).half()
    # print(model.config.seq_length)

    config = get_args()
    print(config)
    config.repo_path = config.repo_path + os.sep + config.lang
    run_multi_process(config)