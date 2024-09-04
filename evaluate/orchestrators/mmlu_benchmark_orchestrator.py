import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
from processors.result_processor import calculate_score, path_to_results

class MMLUBenchmarkOrchestrator:
    def __init__(self, model, tokenizer, benchmark_name, model_name, nshot, project_root):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.nshot = nshot
        self.project_root = project_root
        self.choices = ["A", "B", "C", "D"]

        self.categories = self._import_module('categories', f'benchmarks.benchmarks.{benchmark_name}.code')
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(project_root, f'benchmarks/benchmarks/{benchmark_name}/data')

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
        test_question_directory = os.path.join(self.data_folder_path, 'test')
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(test_question_directory) if "_test.csv" in f])

        all_cors = []
        for subject in subjects:
            example_questions_df = pd.read_csv(os.path.join(self.data_folder_path, "dev", f"{subject}_dev.csv"), header=None)[:self.nshot]
            test_question_df = pd.read_csv(os.path.join(self.data_folder_path, "test", f"{subject}_test.csv"), header=None)

            cors, acc, probs = self._eval_subject(subject, example_questions_df, test_question_df)
            self._save_results(subject, test_question_df, cors, probs)
            
            all_cors.append(cors)

        average_acc = calculate_score(all_cors)
        self._save_score(average_acc)

        print(f"Average accuracy: {average_acc:.3f}")

    def _format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        for j, choice in enumerate(self.choices):
            prompt += f"\n{choice}. {df.iloc[idx, j+1]}"
        prompt += "\nAnswer:"
        if include_answer:
            prompt += f" {df.iloc[idx, 5]}"
        return prompt

    def _format_prompt(self, example_questions_df, test_question_df, test_question_idx):
        prompt = "Answer the following multiple choice questions. Choose the best answer from A, B, C, or D.\n\n"
        for i in range(len(example_questions_df)):
            prompt += self._format_example(example_questions_df, i) + "\n\n"
        prompt += self._format_example(test_question_df, test_question_idx, include_answer=False)
        return prompt

    def _eval_subject(self, subject, example_questions_df, test_question_df):
        cors = []
        preds = []
        probs = []

        for i in range(len(test_question_df)):
            probability, prediction, correctness = self._eval_question(example_questions_df, test_question_df, i)
            probs.append(probability)
            preds.append(prediction)
            cors.append(correctness)

        acc = np.mean(cors)
        print(f"{subject} Accuracy: {acc:.3f}")

        return cors, acc, probs

    def _eval_question(self, example_questions_df, test_question_df, test_question_number):
        prompt = self._format_prompt(example_questions_df, test_question_df, test_question_number)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in self.choices]
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_probs)]
        
        return choice_probs, pred, pred == test_question_df.iloc[test_question_number, 5]

    def _save_results(self, subject, test_question_df, cors, probs, preds):
        results_dir = path_to_results(self.project_root, self.benchmark_name, self.model_name, True)
        os.makedirs(results_dir, exist_ok=True)

        test_question_df["correct"] = cors
        test_question_df["prediction"] = preds
        for j, choice in enumerate(self.choices):
            test_question_df[f"choice{choice}_probs"] = [p[j] for p in probs]

        test_question_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)

    def _save_score(self, average_acc):
        results_dir = path_to_results(self.project_root, self.benchmark_name, self.model_name, False)
        score_file_path = os.path.join(results_dir, f"{self.benchmark_name}_score.txt")
        
        with open(score_file_path, 'w') as f:
            f.write(f"{average_acc:.3f}")
        
        print(f"Score saved to: {score_file_path}")