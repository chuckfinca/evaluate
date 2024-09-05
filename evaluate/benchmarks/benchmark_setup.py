import os
import requests
import zipfile
import tarfile
import shutil
from evaluate.utils.path_utils import get_benchmark_directory
from .benchmark_config import get_benchmark_config

def setup_benchmark(benchmark_name, is_test=False):

    config = get_benchmark_config(benchmark_name)
    if not config:
        raise ValueError(f"Benchmark '{benchmark_name}' is not supported.")
    
    benchmark_path = get_benchmark_directory(benchmark_name, is_test)

    # Download and extract code
    code_path = os.path.join(benchmark_path, 'code')
    os.makedirs(code_path, exist_ok=True)

    if not os.listdir(code_path):  # Check if directory is empty
        _download_and_extract(config['code_url'], code_path, is_zip=True)

    # Download and extract data
    data_path = os.path.join(benchmark_path, 'data')
    os.makedirs(data_path, exist_ok=True)

    if not os.listdir(data_path):  # Check if directory is empty
        _download_and_extract(config['data_url'], data_path, is_zip=False)

    print(f"Benchmark '{benchmark_name}' has been set up successfully.")

def _download_and_extract(url, path, is_zip=True):
    
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # Download the file
    response = requests.get(url)
    file_path = os.path.join(path, 'temp_archive')
    with open(file_path, 'wb') as f:
        f.write(response.content)

    # Extract the contents
    if is_zip:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    else:
        with tarfile.open(file_path, 'r:*') as tar_ref:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                    tar.extract(member, path, set_attrs=False, numeric_owner=numeric_owner)

            safe_extract(tar_ref, path=path)

    # Remove the temporary archive file
    os.remove(file_path)

    # Move contents from top-level folder to the target path
    extracted_items = os.listdir(path)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(path, extracted_items[0])):
        top_level_folder = os.path.join(path, extracted_items[0])
        for item in os.listdir(top_level_folder):
            shutil.move(os.path.join(top_level_folder, item), path)
        os.rmdir(top_level_folder)

    print(f"Extraction completed to: {path}")
