import os
import sys

# Base path for all projects
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Function to add a path to sys.path if it's not already there
def add_to_sys_path(path):
    if path not in sys.path:
        print(f'adding path: {path}')
        sys.path.insert(0, path)

add_to_sys_path(PROJECT_ROOT)
