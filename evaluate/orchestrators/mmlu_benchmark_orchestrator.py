import os
import numpy as np
import pandas as pd
import torch
import csv
from datetime import datetime
from evaluate.processors.result_processor import calculate_scores
from evaluate.utils.import_utils import import_benchmark_module
from evaluate.utils.path_utils import get_benchmark_directory, path_to_results
from evaluate.logs.logger import logger

class MMLUEvaluationOrchestrator:
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.benchmark_name = config['benchmark_name']
        self.model_name = config['model_name']
        self.nshot = config.get('nshot', 0)
        
        # Load prompt template from config
        self.load_prompt_template(config['prompt_template'])
        
        self.choices = ["A", "B", "C", "D"]

        benchmark_path = get_benchmark_directory(self.benchmark_name)
        self.categories = import_benchmark_module('categories', benchmark_path)
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(benchmark_path, 'data')

    def load_prompt_template(self, prompt_template):
        self.prompt_template = prompt_template.get("main_prompt_template", "")
        self.question_template = prompt_template.get("question_template", "")
        self.question_separator = prompt_template.get("question_separator", "\n\n")
        self.instructions_template = prompt_template.get("instructions", "")

    def print_prompt_template(self):
        example_questions = [f"{{example_{i+1}}}" for i in range(self.nshot)]
        formatted_instructions = self.format_instructions()
        return self._format_prompt_template(formatted_instructions, example_questions, "{test question}")

    def evaluate(self):
        logger.log.info("Prompt template:")
        logger.log.info(self.print_prompt_template())
        logger.log.info("Special Tokens:")
        logger.log.info(self.tokenizer.sep_token)
        logger.log.info(self.tokenizer.eos_token)
        logger.log.info(self.tokenizer.all_special_tokens)

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

            cors, probs, preds = self._evaluate_subject(subject, example_questions_df, test_question_df)
            self._save_results(subject, test_question_df, cors, probs, preds)
            
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
        preds = []
        probs = []

        # log the prompt for the first question for each subject as a sanity check
        log_example_prompt = True
        for i in range(len(test_question_df)):
            probability, prediction, correctness = self._evaluate_question(subject, example_questions_df, test_question_df, i, log_example_prompt)
            probs.append(probability)
            preds.append(prediction)
            cors.append(correctness)
            if log_example_prompt:
                log_example_prompt = False

        acc = np.mean(cors)
        logger.log.info(f"{subject} Accuracy: {acc:.3f}")

        return cors, probs, preds

    def _evaluate_question(self, subject, example_questions_df, test_question_df, test_question_number, log_prompt):
        instructions = self.format_instructions(subject.replace("_", " "))
        human_readable_prompt = self._format_prompt(instructions, example_questions_df, test_question_df, test_question_number)
        prompt = "<|begin_of_text|>"
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": human_readable_prompt},
            {"role": "assistant", "content": ""}
        ]
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>" + (f"\n{content}<|eot_id|>" if content else "")

        if log_prompt:
            logger.log.info(f"\n------ prompt ({subject}):")
            logger.log.info(prompt)
            logger.log.info("------")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in self.choices]
        pred = {0: self.choices[0], 1: self.choices[1], 2: self.choices[2], 3: self.choices[3]}[np.argmax(choice_probs)]
        
        return choice_probs, pred, pred == test_question_df.iloc[test_question_number, 5]

    def _format_prompt_template(self, instructions, example_questions, test_question):
        # Start with the original template
        template = self.prompt_template
        
        formatted_example_questions = self.question_separator.join(example_questions)
        
        return template.format(
            instructions=instructions,
            examples=formatted_example_questions,
            question=test_question
        ).strip()
    
    
    def _format_question_template(self, question, choices, answer=None):
        return self.question_template.format(
            question=question.strip(),
            label_a=self.choices[0],
            label_b=self.choices[1],
            label_c=self.choices[2],
            label_d=self.choices[3],
            choice_a=choices[self.choices[0]].strip() if isinstance(choices[self.choices[0]], str) else choices[self.choices[0]],
            choice_b=choices[self.choices[1]].strip() if isinstance(choices[self.choices[1]], str) else choices[self.choices[1]],
            choice_c=choices[self.choices[2]].strip() if isinstance(choices[self.choices[2]], str) else choices[self.choices[2]],
            choice_d=choices[self.choices[3]].strip() if isinstance(choices[self.choices[3]], str) else choices[self.choices[3]],
            answer=answer if answer is not None else ""
        )
    
    def _format_prompt(self, instructions, example_questions_df, test_question_df, test_question_idx):
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
                self.choices[0]: row[1],
                self.choices[1]: row[2],
                self.choices[2]: row[3],
                self.choices[3]: row[4]
            }
        answer = row[5] if include_answer else None
        
        return question, choices, answer

    def format_instructions(self, subject="{subject}"):
        return self.instructions_template.format(
            subject=subject.replace("high_school_","").replace("college_","").replace("elementary_",""),
            label_a=self.choices[0],
            label_b=self.choices[1],
            label_c=self.choices[2],
            label_d=self.choices[3]
        )

    def _save_results(self, subject, test_question_df, cors, probs, preds):
        results_dir = path_to_results(self.benchmark_name, self.model_name, True)
        os.makedirs(results_dir, exist_ok=True)

        test_question_df["correct"] = cors
        test_question_df["prediction"] = preds
        for j, choice in enumerate(self.choices):
            test_question_df[f"choice{choice}_probs"] = [p[j] for p in probs]

        test_question_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)

    def _save_scores(self, macro_avg, micro_avg, subject_results):
        results_dir = path_to_results(self.benchmark_name, self.model_name, False)
        score_file_path = os.path.join(results_dir, f"{self.benchmark_name}_scores.csv")
        
        with open(score_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Benchmark Name', self.benchmark_name])
            writer.writerow(['Model Name', self.model_name])
            writer.writerow(['Evaluation Date', datetime.now().isoformat()])
            writer.writerow(['Macro Average Accuracy', f"{macro_avg:.3f}"])
            writer.writerow(['Micro Average Accuracy', f"{micro_avg:.3f}"])
            writer.writerow(['N-shot', self.nshot])
            writer.writerow(['Prompt Template', self.print_prompt_template()])
            writer.writerow([''])
            writer.writerow(['Subject', 'Accuracy'])
            for subject, accuracy in subject_results.items():
                writer.writerow([subject, f"{accuracy:.3f}"])
        
        logger.log.info(f"Scores saved to: {score_file_path}")