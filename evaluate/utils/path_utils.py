import os
from platformdirs import user_data_dir
from evaluate.utils.module_utils import get_package_name

def get_user_directory(for_tests=False):
    path = user_data_dir(get_package_name(), appauthor=False)
    if for_tests:
        path = os.path.join(path, 'tests')
    return path

def get_benchmark_directory_path(benchmark_name, for_tests=False):
    user_directory = get_user_directory(for_tests)
    benchmark_path = os.path.join(user_directory, 'benchmarks', benchmark_name)
    os.makedirs(benchmark_path, exist_ok=True)
    return benchmark_path