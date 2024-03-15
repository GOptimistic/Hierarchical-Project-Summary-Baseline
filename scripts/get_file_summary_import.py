import re
from random import sample
from transformers import AutoTokenizer, AutoModel
import time
import functools
import signal
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

timeout_seconds = 600

def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator


def get_args():
    parser = argparse.ArgumentParser(
        """Get file summary of each project""")

    parser.add_argument("--node_num", type=int, default=4)
    parser.add_argument("--split_data_num", type=int, default=16)
    parser.add_argument("--data_part_size", type=int, default=1000)
    parser.add_argument("--start_part_index", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="/home/LAB/guanz/gz_graduation/code_embedding_pretrained_model/chatglm3-6b-128k")
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--csv_data_path", type=str, default="/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/import_analyze_all.csv")
    parser.add_argument("--output_data_path", type=str, default="/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/file_summary_data")
    parser.add_argument("--log_path", type=str,
                        default="/home/LAB/guanz/gz_graduation/model_file_summary/scripts/logs")
    parser.add_argument("--max_length_file", type=int, default=50)

    args = parser.parse_args()
    return args


@timeout(timeout_seconds)
def get_single_project_file_summary(repo_name, files_info, model, tokenizer, max_length_file):
    sequence_max_length = model.config.seq_length
    file_summaries = {}
    files_info_object = json.loads(files_info)
    files_info_items = files_info_object.items()
    if len(files_info_items) > max_length_file:
        files_info_items = sorted(files_info_items, key=lambda x: x[1]["degree"], reverse=True)[:max_length_file]
    for file_name, file_info in files_info_items:
        file_path = file_info['path']
        with open(file_path, 'r', encoding='utf-8') as f:
            code_text = f.read()

        question = 'Please help me write a one-sentence summary of the java class file {} from the open-source project named {}. The summary should capture the overall functionality of the file in no longer than 30 words, without describing each function individually. The code is\n\n'.format(
            file_name,
            repo_name
        )
        question = question + code_text
        if len(tokenizer.tokenize(question)) >= sequence_max_length:
            continue
        response, history = model.chat(tokenizer, question, history=[])
        file_summaries[file_name] = response
    return file_summaries

def run_single_process(config, data_list, start_index, node_index):
    start_time = time.time()
    model_path = config.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
    device = "cuda:{}".format(node_index)
    with open(config.log_path + os.sep + '{}_{}.txt'.format(start_index, node_index), 'w') as f:
        print(device, file=f, flush=True)
        model.to(device)
        model = model.eval()
        with open(config.output_data_path + os.sep + 'data_import_file_summary_{}_{}.csv'.format(config.start_part_index, node_index), 'w', newline='', encoding='utf-8') as out_file:
            csv_witer = csv.writer(out_file)
            # head of csv
            csv_witer.writerow(['repo_name', 'file_summaries', 'repo_summary'])

            for i in range(len(data_list)):
                repo_name, files_info, repo_summary = data_list[i]
                try:
                    file_summaries = get_single_project_file_summary(repo_name, files_info, model, tokenizer, config.max_length_file)
                    if len(file_summaries.keys()) == 0:
                        print('part {} node {} index {} is null'.format(config.start_part_index, node_index, i), file=f, flush=True)
                        continue
                    csv_witer.writerow([repo_name, json.dumps(file_summaries), repo_summary])
                except RuntimeError as runtime_exception:
                    if "CUDA out of memory" in str(runtime_exception):
                        print("ERROR: out of memory. part {} node {} index {}".format(config.start_part_index, node_index, i), file=f, flush=True)
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                        continue
                    else:
                        print(
                            "Runtime ERROR: {}. part {} node {} index {}".format(str(runtime_exception), config.start_part_index, node_index, i), file=f, flush=True)
                        continue
                except TimeoutError as timeout_exception:
                    print("Timeout ERROR: {}. part {} node {} index {}".format(str(timeout_exception), config.start_part_index, node_index, i), file=f, flush=True)
                    continue
                except Exception as e:
                    print("Unknown ERROR: {}. part {} node {} index {}".format(str(e), config.start_part_index, node_index, i), file=f, flush=True)
                    continue

                print('part {} node {} index {} done'.format(config.start_part_index, node_index, i), file=f, flush=True)

        end_time = time.time()
        print('###### Process {} has done. Use {}s'.format(node_index, end_time - start_time), file=f, flush=True)


def run_multi_process(config):
    repos_and_summaries = []
    index = 0
    with open(config.csv_data_path, 'r', encoding='utf-8') as in_file:
        csv_reader = csv.reader(in_file)
        for row in csv_reader:
            if index == 0:
                index = index + 1
                continue
            repo_name = row[0]
            files_info = row[1]
            repo_summary = row[2]
            repos_and_summaries.append((repo_name, files_info, repo_summary))
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
        processes.append(Process(target=run_single_process, args=(config, repos_and_summaries[start_index:end_index], config.start_part_index, i,)))
    [p.start() for p in processes]  # 开启进程
    [p.join() for p in processes]  # 等待进程依次结束


if __name__ == "__main__":
    config = get_args()
    print(config)
    run_multi_process(config)