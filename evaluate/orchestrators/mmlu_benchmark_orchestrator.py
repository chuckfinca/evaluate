import os
import numpy as np
import pandas as pd
import torch
from evaluate.processors.result_processor import calculate_score, path_to_results
from evaluate.utils.module_utils import import_benchmark_module
from evaluate.utils.path_utils import get_benchmark_directory_path

class MMLUBenchmarkOrchestrator:
    
    prompt_template = """
{instructions}
{example_questions}
{test_question}
""".strip()
        
    question_template = """
{question}
(A) {choice_a} (B) {choice_b} (C) {choice_c} (D) {choice_d}
Answer:{answer}
""".strip()
    
    def __init__(self, model, tokenizer, benchmark_name, model_name, nshot):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.nshot = nshot
        self.choices = ["A", "B", "C", "D"]

        benchmark_path = get_benchmark_directory_path(benchmark_name)
        self.categories = import_benchmark_module('categories', benchmark_path)
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(benchmark_path, 'data')

    def evaluate(self):
        test_question_directory = os.path.join(self.data_folder_path, 'test')
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(test_question_directory) if "_test.csv" in f])

        all_cors = []
        for subject in subjects:
            example_questions_df = pd.read_csv(os.path.join(self.data_folder_path, "dev", f"{subject}_dev.csv"), header=None)[:self.nshot]
            test_question_df = pd.read_csv(os.path.join(self.data_folder_path, "test", f"{subject}_test.csv"), header=None)

            cors, probs, preds = self._eval_subject(subject, example_questions_df, test_question_df)
            self._save_results(subject, test_question_df, cors, probs, preds)
            
            all_cors.append(cors)

        average_acc = calculate_score(all_cors)
        self._save_score(average_acc)

        print(f"Average accuracy: {average_acc:.3f}")

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

        return cors, probs, preds

    def _eval_question(self, example_questions_df, test_question_df, test_question_number, ):
        prompt = self._format_prompt(example_questions_df, test_question_df, test_question_number)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in self.choices]
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_probs)]
        
        return choice_probs, pred, pred == test_question_df.iloc[test_question_number, 5]
    
    def _format_prompt_template(self, instructions, example_questions, test_question):
        # Start with the original template
        template = self.prompt_template
        
        # Handle empty instructions
        if not instructions:
            template = template.replace("{instructions}\n", "")

        # Handle empty example questions
        if not example_questions:
            template = template.replace("{example_questions}\n", "")

        # Format example questions if they exist
        if example_questions:
            formatted_examples = example_questions[0] + "\n" + "\n".join(example_questions[1:])
        else:
            formatted_examples = ""

        return template.format(
            instructions=instructions,
            example_questions=formatted_examples,
            test_question=test_question
        ).strip()
    
    def _format_question_template(self, question, choices, answer=None):
        return self.question_template.format(
            question=question,
            choice_a=choices['A'],
            choice_b=choices['B'],
            choice_c=choices['C'],
            choice_d=choices['D'],
            answer= " " + answer if answer is not None else ""
        )
    
    def _format_prompt(self, example_questions_df, test_question_df, test_question_idx):
        instructions = ""
        example_prompts = []
        for i in range(len(example_questions_df)):
            example_prompts.append(self._format_question(example_questions_df, i, True))

        test_question_prompt = self._format_question(test_question_df, test_question_idx, False)
        
        return self._format_prompt_template(instructions, example_prompts, test_question_prompt)
    
    def _format_question(self, df, row_index, include_answer):
        question, choices, answer = self._process_question_row(df, row_index, include_answer)
        return self._format_question_template(question, choices, answer)

    def _process_question_row(self, df, row_index, include_answer=True):
        row = df.iloc[row_index]
        
        question = row[0]
        choices = {
                'A': row[1],
                'B': row[2],
                'C': row[3],
                'D': row[4]
            }
        answer = row[5] if include_answer else None
        
        return question, choices, answer

    def _save_results(self, subject, test_question_df, cors, probs, preds):
        results_dir = path_to_results(self.benchmark_name, self.model_name, True)
        os.makedirs(results_dir, exist_ok=True)

        test_question_df["correct"] = cors
        test_question_df["prediction"] = preds
        for j, choice in enumerate(self.choices):
            test_question_df[f"choice{choice}_probs"] = [p[j] for p in probs]

        test_question_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)

    def _save_score(self, average_acc):
        results_dir = path_to_results(self.benchmark_name, self.model_name, False)
        score_file_path = os.path.join(results_dir, f"{self.benchmark_name}_score.txt")
        
        with open(score_file_path, 'w') as f:
            f.write(f"{average_acc:.3f}")
        
        print(f"Score saved to: {score_file_path}")