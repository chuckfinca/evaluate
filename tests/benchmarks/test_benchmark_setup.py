import unittest
import os
import shutil
from evaluate.evaluators.evaluator_mmlu import MMLUEvaluator

class TestBenchmarkSetup(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_path = os.path.join(self.base_path, 'test_benchmarks')
        os.makedirs(self.test_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_path)

    def test_mmlu_setup(self):
        # Run the setup
        setup_benchmark('mmlu', self.test_path)

        # Check if the code directory exists and is not empty
        code_path = os.path.join(self.test_path, 'benchmarks', 'benchmarks', 'mmlu', 'code')
        self.assertTrue(os.path.exists(code_path))
        self.assertTrue(len(os.listdir(code_path)) > 0)

        # Check if the data directory exists and is not empty
        data_path = os.path.join(self.test_path, 'benchmarks', 'benchmarks', 'mmlu', 'data')
        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(len(os.listdir(data_path)) > 0)

        # Check for specific files or directories that should be present
        # (You may want to adjust these based on the exact structure of the MMLU benchmark)
        self.assertTrue(os.path.exists(os.path.join(data_path, 'test')))
        self.assertTrue(os.path.exists(os.path.join(data_path, 'dev')))
        self.assertTrue(os.path.exists(os.path.join(data_path, 'val')))

if __name__ == '__main__':
    unittest.main()