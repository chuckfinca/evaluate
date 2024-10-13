import importlib
import json
import os


def get_package_name():
    return __package__.split('.')[0]

def import_benchmark_module(module_name, benchmark_path):
    module_path = os.path.join(benchmark_path, 'code', f'{module_name}.py')
    
    if not os.path.isfile(module_path):
        raise ImportError(f"No module named '{module_name}' in '{benchmark_path}'")
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)