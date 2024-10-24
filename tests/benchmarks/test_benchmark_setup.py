import unittest
import os
import shutil
from finca.evaluate.benchmarks.benchmark_setup import setup_benchmark
from finca.utils.path_utils import path_to_benchmarks, path_to_package_data

class TestBenchmarkSetup(unittest.TestCase):
    def setUp(self):
        self.test_path = path_to_package_data(for_tests=True)
        os.makedirs(self.test_path, exist_ok=True)

    def tearDown(self):
        print("Cleaning up...")
        shutil.rmtree(self.test_path)

    def test_mmlu_setup(self):
        # Run the setup
        print('Setting up mmlu...')
        setup_benchmark('mmlu', is_test=True)

        print('Checking that code was fetched...')
        # Check if the code directory exists and is not empty
        benchmark_path = path_to_benchmarks('mmlu', for_tests=True)
        code_path = os.path.join(benchmark_path, 'code')
        self.assertTrue(os.path.exists(code_path))
        self.assertTrue(len(os.listdir(code_path)) > 0)

        print('Checking that data was fetched...')
        # Check if the data directory exists and is not empty
        data_path = os.path.join(benchmark_path, 'data')
        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(len(os.listdir(data_path)) > 0)

        # Check for specific files or directories that should be present
        self.assertTrue(os.path.exists(os.path.join(data_path, 'test')))
        self.assertTrue(os.path.exists(os.path.join(data_path, 'dev')))

if __name__ == '__main__':
    unittest.main()