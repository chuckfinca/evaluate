import os
from platformdirs import user_data_dir
from finca.utils.import_utils import get_package_name

def path_to_package_data(for_tests=False):
    path = user_data_dir(get_package_name(), appauthor=False)
    if for_tests:
        path = os.path.join(path, 'tests')
    return path

def path_to_benchmarks(benchmark_name, for_tests=False):
    package_data_directory = path_to_package_data(for_tests)
    benchmark_path = os.path.join(package_data_directory, 'benchmarks', benchmark_name)
    os.makedirs(benchmark_path, exist_ok=True)
    return benchmark_path

def path_to_results(benchmark_name, model_name):
    benchmark_path = path_to_benchmarks(benchmark_name)
    
    # Base path for the benchmark data
    path = os.path.join(benchmark_path, "results", model_name)
    os.makedirs(path, exist_ok=True)
    return path

def path_to_raw_results(benchmark_name, model_name, epoch_time):
    results_path = path_to_results(benchmark_name, model_name)
    
    path = os.path.join(results_path, 'raw', f'{epoch_time}')
    os.makedirs(path, exist_ok=True)
    return path