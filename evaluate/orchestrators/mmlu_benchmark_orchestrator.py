import os
import sys
import importlib
import numpy as np
import pandas as pd

from visualizers.chart_creator import create_mmlu_comparison_chart
from processors.result_processor import path_to_results

class MMLUBenchmarkOrchestrator:
    def __init__(self, model, tokenizer, args, project_root):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.categories = self._import_module('categories', f'benchmarks.benchmarks.{args.benchmark_name}.code')
        self.evaluate_module = self._import_module('evaluate_causal_lm', 'benchmarks.custom_evaluators')
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(project_root, f'benchmarks/benchmarks/{args.benchmark_name}/data')

    def _import_module(self, module_name, module_directory):
        try:
            # Construct the full module path
            full_module_path = f'{module_directory}.{module_name}'

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

        average_acc = self.calculate_score(cors)
        print(f"Average accuracy: {average_acc:.3f}")

        return average_acc

    def _eval_subject(self, subject, dev_df, test_df):
        # Use the evaluation logic from the dynamically imported module
        cors, acc, probs = self.evaluate_module.eval(self.args, subject, self.model, self.tokenizer, dev_df, test_df)
        return cors, acc, probs

    def _save_results(self, subject, test_df, cors, probs):
        results_dir = path_to_results(self.args, True)
        os.makedirs(results_dir, exist_ok=True)

        test_df["correct"] = cors
        for j, choice in enumerate(self.evaluate_module.choices):
            test_df[f"choice{choice}_probs"] = [p[j] for p in probs]

        test_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)