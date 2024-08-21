import os
import sys
import importlib
import numpy as np
import pandas as pd

class MMLUEvaluator:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.evaluation_name = args.evaluation

        # Dynamically import the required modules
        self.categories = self._import_module(f'benchmarks.code.{self.evaluation_name}.categories')
        self.evaluate_module = self._import_module(f'benchmarks.code.{self.evaluation_name}.evaluate_flan')

    def _import_module(self, module_path):
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            print(f"Error importing module {module_path}: {e}")
            sys.exit(1)

    def evaluate(self):
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(self.args.data_dir, "test")) if "_test.csv" in f])

        all_cors = []
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(self.args.data_dir, "dev", f"{subject}_dev.csv"), header=None)[:self.args.ntrain]
            test_df = pd.read_csv(os.path.join(self.args.data_dir, "test", f"{subject}_test.csv"), header=None)

            cors, acc, probs = self._eval_subject(subject, dev_df, test_df)
            all_cors.append(cors)

            self._save_results(subject, test_df, cors, probs)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"Average accuracy: {weighted_acc:.3f}")
        return weighted_acc

    def _eval_subject(self, subject, dev_df, test_df):
        # Use the evaluation logic from the dynamically imported module
        cors, acc, probs = self.evaluate_module.eval(self.args, subject, self.model, dev_df, test_df)
        return cors, acc, probs

    def _save_results(self, subject, test_df, cors, probs):
        results_dir = os.path.join(self.args.save_dir, f"results_{self.args.model}")
        os.makedirs(results_dir, exist_ok=True)

        test_df[f"{self.args.model}_correct"] = cors
        for j, choice in enumerate(self.evaluate_module.choices):
            test_df[f"{self.args.model}_choice{choice}_probs"] = probs[:, j]

        test_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)