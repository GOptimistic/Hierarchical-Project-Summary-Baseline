import time
import functools
import signal
import json
import argparse
from multiprocessing import Process
import csv
import sys
import javalang
import os
import networkx as nx

csv.field_size_limit(sys.maxsize)


# 读取mini_all_with_name.csv对每一个仓库进行依赖关系分析，得到被依赖次数最多的前n个文件，然后将这些文件的package+class名，file_path和方法签名列表存在一个json中
# 一共17714条数据，每个用户最多80个CPU,占四个节点，每个节点跑9个进程，一共36个进程，每个进程处理500条数据

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

    parser.add_argument("--node_num", type=int, default=9)
    parser.add_argument("--split_data_num", type=int, default=36)
    parser.add_argument("--data_part_size", type=int, default=500)
    parser.add_argument("--start_part_index", type=int, default=0)
    parser.add_argument("--lang", type=str, default="java")
    parser.add_argument("--csv_data_path", type=str, default="/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/mini_all_with_name.csv")
    parser.add_argument("--output_data_path", type=str, default="/home/LAB/guanz/gz_graduation/model_file_summary/analyze_import_data/part_data")
    parser.add_argument("--repo_path", type=str, default="/home/LAB/guanz/gz_graduation/clone_github_repo_data/github_repo_data")
    parser.add_argument("--max_length_file", type=int, default=100)

    args = parser.parse_args()
    return args


def should_skip_file(file_name):
    """
    根据文件路径判断是否应该跳过该文件。
    """
    skip_keywords = ['util', 'test', 'doc', 'tmp', 'temp', 'backup', 'old', 'demo', 'example', 'archive']
    for keyword in skip_keywords:
        if keyword in file_name.lower():
            return True
    return False


def parse_package_and_imports(file_path):
    """
    解析Java文件，尝试提取包名和导入的依赖。
    在解析过程中遇到错误则跳过该文件。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            full_content = ''
            for d in source:
                # 需要剔除 // 开头的日志
                if not d.strip().startswith("//"):
                    full_content += d
        tree = javalang.parse.parse(full_content)
        package_name = None
        imports = set()
        for _, node in tree.filter(javalang.tree.PackageDeclaration):
            package_name = node.name
        for _, node in tree.filter(javalang.tree.Import):
            imports.add(node.path)

        method_signatures = []
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            params = ', '.join([f"{param.type.name} {param.name}" for param in node.parameters])
            method_signature = f"{node.return_type.name if node.return_type else 'void'} {node.name}({params})"
            method_signatures.append(method_signature)
        if len(method_signatures) > 0:
            return package_name, imports, method_signatures
        else:
            return None, set(), []
    except javalang.parser.JavaSyntaxError:
        # print(f"Syntax error in file: {file_path}. File will be skipped.")
        return None, set(), []
    except Exception:
        # print(f"Error parsing file: {file_path}. Error: {e}. File will be skipped.")
        return None, set(), []


def build_project_structure(root_dir):
    """
    构建项目结构，记录每个文件的包名，并分析依赖关系。
    """
    package_file_map = {}
    file_package_map = {}
    package_imports_map = {}
    package_methods_map = {}
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.java') and not should_skip_file(file):
                file_path = os.path.join(subdir, file)
                package_name, imports, method_signatures = parse_package_and_imports(file_path)
                if package_name:
                    class_name = file.replace('.java', '')
                    full_name = f"{package_name}.{class_name}"
                    package_file_map[full_name] = file_path
                    file_package_map[file_path] = full_name
                    package_imports_map[full_name] = imports
                    package_methods_map[full_name] = method_signatures
    return package_file_map, file_package_map, package_imports_map, package_methods_map


def build_dependency_graph(package_imports_map, package_file_map):
    """
    构建只包含内部依赖的依赖图。
    """
    graph = nx.DiGraph()

    for package_name, imports in package_imports_map.items():
        for import_path in imports:
            if import_path in package_file_map:
                graph.add_edge(package_name, import_path)
    return graph


def get_top_dependent_classes(graph, n):
    """
    返回被依赖次数最多的n个类的名称。
    """
    return sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)[:n]


@timeout(timeout_seconds)
def analyze_single_project(root_dir, max_file_num):
    package_file_map, file_package_map, package_imports_map, package_methods_map = build_project_structure(root_dir)
    graph = build_dependency_graph(package_imports_map, package_file_map)

    files_info = {}
    top_classes = get_top_dependent_classes(graph, max_file_num)
    for class_info in top_classes:
        class_name, degree = class_info
        method_signatures = package_methods_map[class_name]
        file_path = package_file_map[class_name]
        single_file_info = {
            "in_degree": degree,
            "file_path": file_path,
            "method_signatures": method_signatures
        }
        files_info[class_name] = single_file_info

    return files_info

def run_single_process(config, data_list, node_index):
    start_time = time.time()
    with open(config.output_data_path + os.sep + 'data_{}_import_analyze_{}_{}.csv'.format(config.lang, config.start_part_index, node_index), 'w', newline='') as out_file:
        csv_witer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # head of csv
        csv_witer.writerow(['repo_name', 'files_info', 'repo_summary'])

        for i in range(len(data_list)):
            data = data_list[i]
            full_name = data[0]
            summary = data[1]
            project_path = config.repo_path + os.sep + full_name.replace('/', '_')
            try:
                files_info = analyze_single_project(project_path, config.max_length_file)
                if len(files_info.keys()) == 0:
                    print('part {} node {} index {} is null'.format(config.start_part_index, node_index, i))
                    continue
                csv_witer.writerow([full_name, json.dumps(files_info), summary])
            except TimeoutError as timeout_exception:
                print("Timeout ERROR: {}. part {} node {} index {}".format(str(timeout_exception), config.start_part_index, node_index, i))
                continue
            except Exception as e:
                print("Unknown ERROR: {}. part {} node {} index {}".format(str(e), config.start_part_index, node_index, i))
                continue

            print('part {} node {} index {} done'.format(config.start_part_index, node_index, i))

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
            full_name = row[0]
            summary = row[1]
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
    config = get_args()
    print(config)
    config.repo_path = config.repo_path + os.sep + config.lang
    run_multi_process(config)