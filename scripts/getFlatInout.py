import csv
import json
import sys
import os
from random import sample
import re

csv.field_size_limit(sys.maxsize)

src_csv_data_path = '../data/mini_all_with_name.csv'
dst_csv_data_path = '../data/mini_all_flat.csv'
repo_path = '/home/LAB/guanz/gz_graduation/clone_github_repo_data/github_repo_data/java'
max_length_package = 5
max_length_file = 5
max_length_method = 5
filterout_regex = 'util|test|docs'
keywords = ['if', 'for', 'while', 'try', 'catch', 'throw', 'else', 'switch', 'case', 'do']


def get_method_names(code_text):
    # 正则表达式匹配Java函数声明，忽略访问修饰符
    # \b 表示单词边界，确保精确匹配单词
    # [\w<>\[\]]+ 匹配返回类型，包括泛型和数组类型
    # (\w+) 捕获函数名
    # \s*\([^)]*\) 匹配参数列表，包括任何非)的字符
    pattern = r'\b[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*{'

    matches = re.findall(pattern, code_text)
    function_names = []
    for match in matches:
        if match.lower() not in keywords:
            function_names.append(match)
    return function_names


def get_project_flat_input(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        text = f.read()
    project = json.loads(text)
    packages = project["packages"]
    project_name = project["full_name"].split('/')[1]
    flat_input = 'project {} '.format(project_name)
    package_keys = packages.keys()
    if len(package_keys) > max_length_package:
        package_keys = sample(package_keys, max_length_package)
    for package in package_keys:
        if len(re.findall(filterout_regex, package, re.IGNORECASE)) != 0:
            continue
        files = packages[package]['files']
        files_keys = files.keys()
        package_info = 'package {} '.format(package)
        if len(files_keys) > max_length_file:
            files_keys = sample(files_keys, max_length_file)
        for file in files_keys:
            if len(re.findall(filterout_regex, file, re.IGNORECASE)) != 0:
                continue
            methods = files[file]['methods']
            if len(methods) > max_length_method:
                methods = sample(methods, max_length_method)
            methods_name = []
            for method in methods:
                methods_name.extend(get_method_names(method))
            file_info = 'file {} method {} '.format(file, ' '.join(methods_name))
            package_info += file_info
        flat_input += package_info

    return flat_input


with open(src_csv_data_path, 'r') as in_file, open(dst_csv_data_path, 'w', newline='') as out_file:
    csv_reader = csv.reader(in_file)
    csv_witer = csv.writer(out_file)
    index = 0
    for row in csv_reader:
        print(index)
        if index == 0:
            row.append('flat_input')
            csv_witer.writerow(row)
            index = index + 1
            continue
        repo_name = row[0]
        repo_name = repo_name.replace('/', '_')
        json_path = repo_path + os.sep + repo_name.replace('/', '_') + '/repo_tree_info.json'
        flat_input = get_project_flat_input(json_path)
        row.append(flat_input)
        csv_witer.writerow(row)
        index = index + 1
print('###### Done')