import os
import sys

# Base path for all projects
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Project-specific paths
MMLU_PATH = os.path.join(PROJECT_ROOT, 'benchmarks', 'benchmarks', 'mmlu')
# OTHER_PROJECT_PATH = os.path.join(PROJECT_ROOT, 'other_project')

# You can add more paths as needed
PATHS = {
    'mmlu': MMLU_PATH,
    # 'other_project': OTHER_PROJECT_PATH,
    # Add more projects here
}

def get_evaluation_project_path(evaluation_name):
    return PATHS.get(evaluation_name)

# Function to add a path to sys.path if it's not already there
def add_to_sys_path(path):
    if path not in sys.path:
        print(f'adding path: {path}')
        sys.path.insert(0, path)

add_to_sys_path(PROJECT_ROOT)
