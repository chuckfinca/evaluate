import os
from platformdirs import user_data_dir
from evaluate.utils.import_utils import get_package_name

def get_package_data_directory(for_tests=False):
    path = user_data_dir(get_package_name(), appauthor=False)
    if for_tests:
        path = os.path.join(path, 'tests')
    return path

def get_benchmark_directory(benchmark_name, for_tests=False):
    package_data_directory = get_package_data_directory(for_tests)
    benchmark_path = os.path.join(package_data_directory, 'benchmarks', benchmark_name)
    os.makedirs(benchmark_path, exist_ok=True)
    return benchmark_path

def path_to_results(benchmark_name, model_name, raw):
    benchmark_path = get_benchmark_directory(benchmark_name)
    
    # Base path for the benchmark data
    path = os.path.join(benchmark_path, "results", model_name)
    if raw:
        path = os.path.join(path, 'raw')

    os.makedirs(path, exist_ok=True)
    return path