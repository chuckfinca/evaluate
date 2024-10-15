import os
import time
import numpy as np
import pandas as pd
import torch
import csv
import re
from finca.evaluate.processors.result_processor import calculate_scores
from finca.utils.import_utils import import_benchmark_module
from finca.utils.path_utils import path_to_benchmarks, path_to_raw_results, path_to_results
from finca.logs.logger import logger
from finca.prompt_managers.prompt_manager_factory import PromptManagerFactory

class MMLUEvaluationOrchestrator:
    
    def __init__(self, model, tokenizer, prompt_manager, config):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_manager = prompt_manager
        self.config = config
        
        self.benchmark_name = config['benchmark_name']
        self.model_name = config['model_name']
        self.nshot = config.get('nshot', 0)
        self.generation_type = config.get('generation_type', 'inference')
        
        
        self.choices = config['answer_choices']

        # by default we will log the prompt for the first question for each subject as a sanity check
        self.log_prompt = True

        benchmark_path = path_to_benchmarks(self.benchmark_name)
        self.categories = import_benchmark_module('categories', benchmark_path)
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(benchmark_path, 'data')
        
        self.results_dir = path_to_results(self.benchmark_name, self.model_name)
        self.raw_results_path = path_to_raw_results(self.benchmark_name, self.model_name, int(time.time()))

    def evaluate(self):
        logger.log.info("Prompt template:")
        logger.log.info(self.prompt_manager.print_prompt())

        test_question_directory = os.path.join(self.data_folder_path, 'test')
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(test_question_directory) if "_test.csv" in f])

        all_cors = []
        all_subject_accs = []
        subject_results = {}
        
        if self.config['cap_subjects']:
            subjects = subjects[:7]

        for subject in subjects:
            example_questions_df = pd.read_csv(os.path.join(self.data_folder_path, "dev", f"{subject}_dev.csv"), header=None)[:self.nshot]
            test_question_df = pd.read_csv(os.path.join(self.data_folder_path, "test", f"{subject}_test.csv"), header=None)

            cors = self._evaluate_subject(subject, example_questions_df, test_question_df)
            
            all_cors.extend(cors)
            subject_acc = np.mean(cors)
            all_subject_accs.append(subject_acc)
            subject_results[subject] = subject_acc

        macro_avg, micro_avg = calculate_scores(all_subject_accs, all_cors)
        self._save_scores(macro_avg, micro_avg, subject_results)

        logger.log.info(f"Macro average accuracy: {macro_avg:.3f}")
        logger.log.info(f"Micro average accuracy: {micro_avg:.3f}")

    def _evaluate_subject(self, subject, example_questions_df, test_question_df):
        cors = []
        
        self.log_prompt = True

        for i in range(len(test_question_df)):
            correctness = self._evaluate_question(subject, example_questions_df, test_question_df, i)
            cors.append(correctness)
            if self.log_prompt:
                self.log_prompt = False

        acc = np.mean(cors)
        logger.log.info(f"{subject} Accuracy: {acc:.3f}")

        return cors
    
    def _evaluate_question(self, subject, example_questions_df, test_question_df, test_question_number):
        question = test_question_df.iloc[test_question_number]
        prompt = self.prompt_manager.prepare_prompt(subject, example_questions_df, question)
        
        if self.generation_type == "open_ended":
            pred = self._open_ended_generation(prompt)
        else:
            pred = self._inference(prompt)

        correct_answer = self._correct_answer(question)

        # Log the inference result
        self._log_inference_result(subject, prompt, test_question_df, test_question_number, {}, pred, correct_answer)

        is_correct = pred == correct_answer
        if self.log_prompt:
            logger.log.info(f"\n------ prompt ({subject}):")
            logger.log.info(prompt)
            logger.log.info(f"pred: {pred}")
            logger.log.info(f"correct_answer: {correct_answer}")
            logger.log.info("------")
        
        return is_correct

    def _open_ended_generation(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id = self.tokenizer.eos_token_id,
                max_new_tokens=50,
                do_sample=False,  # This is all you need for pure greedy decoding (i.e. it will deterministically pick the most likely token)
                temperature=None, # required for do_sample=False
                top_p=None # required for do_sample=False
            )
        
        # Extract the actual answer from the generated text
        generated_answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
        return self._extract_letter(generated_answer)
    
    def _inference(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in self.choices]
        pred = {0: self.choices[0], 1: self.choices[1], 2: self.choices[2], 3: self.choices[3]}[np.argmax(choice_probs)]
        
        return pred
    
    def _correct_answer(self, question_df):
        return question_df[5]

    def _extract_letter(self, text):
        pattern = r'\b([ABCD])\b(?:\s|$|\.)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    def _save_scores(self, macro_avg, micro_avg, subject_results):
        score_file_path = os.path.join(self.results_dir, f"{self.benchmark_name}_scores.csv")
        
        with open(score_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Benchmark Name', self.benchmark_name])
            writer.writerow(['Model Name', self.model_name])
            writer.writerow(['Evaluation Date', time.time()])
            writer.writerow(['Macro Average Accuracy', f"{macro_avg:.3f}"])
            writer.writerow(['Micro Average Accuracy', f"{micro_avg:.3f}"])
            writer.writerow(['N-shot', self.nshot])
            writer.writerow(['Prompt Template', self.prompt_manager.print_prompt()])
            writer.writerow([''])
            writer.writerow(['Subject', 'Accuracy'])
            for subject, accuracy in subject_results.items():
                writer.writerow([subject, f"{accuracy:.3f}"])
        
        logger.log.info(f"Scores saved to: {score_file_path}")
        
    def _log_inference_result(self, subject, prompt, test_question_df, test_question_number, choice_probs, pred, correct_answer):
        log_file_path = os.path.join(self.raw_results_path, f"{self.generation_type}_log.csv")
        
        # Prepare the row data
        row_data = {
            "timestamp": time.time(),
            "subject": subject,
            "question_number": test_question_number,
            "question": test_question_df.iloc[test_question_number, 0],
            "prompt": prompt,
            "choice_A": test_question_df.iloc[test_question_number, 1],
            "choice_B": test_question_df.iloc[test_question_number, 2],
            "choice_C": test_question_df.iloc[test_question_number, 3],
            "choice_D": test_question_df.iloc[test_question_number, 4],
            "correct_answer": correct_answer,
            "predicted_answer": pred,
            "is_correct": pred == correct_answer
        }

        if choice_probs:
            row_data["prob_A"] = choice_probs[0]
            row_data["prob_B"] = choice_probs[1]
            row_data["prob_C"] = choice_probs[2]
            row_data["prob_D"] = choice_probs[3]

        # Check if the file exists to determine whether to write headers
        file_exists = os.path.isfile(log_file_path)
        
        with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=row_data.keys())
            
            if not file_exists:
                writer.writeheader()  # Write header if the file is newly created
            
            writer.writerow(row_data)