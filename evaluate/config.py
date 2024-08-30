import os
import sys

# Base path for all projects
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Function to add a path to sys.path if it's not already there
def add_to_sys_path(path):
    if path not in sys.path:
        print(f'adding path: {path}')
        sys.path.insert(0, path)

def get_project_root():
    """
    Determines the project root directory path.
    Works both locally and in Google Colab.
    """
    try:
        # Check if running in Colab
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        # In Colab, use the current working directory as the project root
        return os.getcwd()
    else:
        # Locally, traverse up until we find a specific file or directory that indicates the project root
        current_path = os.path.abspath(__file__)
        while current_path != '/':
            if os.path.exists(os.path.join(current_path, 'main.py')):  # You can change this to any file that's always in your project root
                return current_path
            current_path = os.path.dirname(current_path)
        
        # If we couldn't find the project root, use the current working directory
        return os.getcwd()

# Example usage
project_root = get_project_root()
print(f"Project root: {project_root}")
print(f"vs: {PROJECT_ROOT}")

add_to_sys_path(PROJECT_ROOT)
