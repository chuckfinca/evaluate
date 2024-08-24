import os
import pprint
import sys
import importlib
import numpy as np
import pandas as pd

class MMLUEvaluator:
    def __init__(self, model, tokenizer, args, project_root):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.benchmark_name = args.benchmark_name

        # Base path for the benchmark code
        self.code_module_path = f'benchmarks.benchmarks.{self.benchmark_name}.code'

        self.categories = self._import_module('categories')
        self.evaluate_module = self._import_module('evaluate_flan')
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(project_root, f'benchmarks/benchmarks/{self.benchmark_name}/data')

    def _import_module(self, module_name):
        try:
            # Construct the full module path
            full_module_path = f'{self.code_module_path}.{module_name}'
            
            # Import the module
            module = importlib.import_module(full_module_path)

            sys.modules[module_name] = module
            
            return module
        except ImportError as e:
            print(f"Error importing module {full_module_path}: {e}")
            sys.exit(1)

    def evaluate(self):
        test_directory = os.path.join(self.data_folder_path, 'test')
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(test_directory) if "_test.csv" in f])

        all_cors = []
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(self.data_folder_path, "dev", f"{subject}_dev.csv"), header=None)[:self.args.ntrain]
            test_df = pd.read_csv(os.path.join(self.data_folder_path, "test", f"{subject}_test.csv"), header=None)

            cors, acc, probs = self._eval_subject(subject, dev_df, test_df)
            all_cors.append(cors)

            self._save_results(subject, test_df, cors, probs)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"Average accuracy: {weighted_acc:.3f}")
        return weighted_acc

    def _eval_subject(self, subject, dev_df, test_df):
        # Use the evaluation logic from the dynamically imported module
        cors, acc, probs = self.evaluate_module.eval(self.args, subject, self.model, self.tokenizer, dev_df, test_df)
        return cors, acc, probs

    def _save_results(self, subject, test_df, cors, probs):
        results_dir = os.path.join(self.args.save_dir, f"results_{self.args.model}")
        os.makedirs(results_dir, exist_ok=True)

        test_df[f"{self.args.model}_correct"] = cors
        for j, choice in enumerate(self.evaluate_module.choices):
            test_df[f"{self.args.model}_choice{choice}_probs"] = probs[:, j]

        test_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)