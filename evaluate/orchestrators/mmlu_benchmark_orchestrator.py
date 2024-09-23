import os
import time
import numpy as np
import pandas as pd
import torch
import csv
import re
from evaluate.processors.result_processor import calculate_scores
from evaluate.utils.import_utils import import_benchmark_module
from evaluate.utils.path_utils import path_to_benchmarks, path_to_raw_results, path_to_results
from evaluate.logs.logger import logger

class MMLUEvaluationOrchestrator:
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.benchmark_name = config['benchmark_name']
        self.model_name = config['model_name']
        self.nshot = config.get('nshot', 0)
        self.generation_type = config.get('generation_type', 'inference')
        
        # format_model_prompt or add_model_specific_instructions
        # Load prompt template from config
        self.load_prompt_template(config['prompt_template'])
        self.review_prompt = config['review_prompt_template'].get("template", "")
        self.system_prompt = config.get('system_prompt', "")
        self.structure_prompt_for_model_input = config.get('structure_prompt_for_model_input', False)
        
        self.choices = ["A", "B", "C", "D"]

        # by default we will log the prompt for the first question for each subject as a sanity check
        self.log_prompt = True

        benchmark_path = path_to_benchmarks(self.benchmark_name)
        self.categories = import_benchmark_module('categories', benchmark_path)
        
        # Base path for the benchmark data
        self.data_folder_path = os.path.join(benchmark_path, 'data')
        
        self.results_dir = path_to_results(self.benchmark_name, self.model_name)
        self.raw_results_path = path_to_raw_results(self.benchmark_name, self.model_name, int(time.time()))

    def load_prompt_template(self, prompt_template):
        self.prompt_template = prompt_template.get("template", "")
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

        return cors, probs, preds

    def _evaluate_question(self, subject, example_questions_df, test_question_df, test_question_number):
        instructions = self.format_instructions(subject.replace("_", " "))
        user_message = self._format_prompt(instructions, example_questions_df, test_question_df, test_question_number)
        
        if self.structure_prompt_for_model_input:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ""},
            ]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            prompt = user_message
        
        if self.generation_type == "open_ended":
            pred = self._open_ended_generation(subject, prompt, test_question_df, test_question_number)
        else:
            pred = self._inference(subject, prompt, test_question_df, test_question_number)
        
        correct_answer = self._correct_answer(test_question_df, test_question_number)

        # Log the inference result
        self._log_inference_result(subject, prompt, test_question_df, test_question_number, None, pred, correct_answer)

        is_correct = pred == correct_answer
        if self.log_prompt:
            logger.log.info(f"\n------ prompt ({subject}):")
            logger.log.info(prompt)
            logger.log.info (f"is correct? {is_correct}")
            logger.log.info("------")

    def _open_ended_generation(self, subject, prompt, test_question_df, test_question_number):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # This is all you need for pure greedy decoding (it will pick the most likely token)
                temperature=None, # required for do_sample=False
                top_p=None # required for do_sample=False
            )
        
        # Extract the actual answer from the generated text
        # generated_answer = generated_answer.split("assistant")[-1].strip()
        generated_answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
        if self.log_prompt:
            logger.log.info(f"\n------ open ended prompt ({subject}):")
            logger.log.info(prompt)
            logger.log.info("------ generated_answer:")
            logger.log.info(generated_answer)
            logger.log.info("------")
            
        return self._extract_letter(generated_answer)
    
    def _inference(self, subject, prompt, test_question_df, test_question_number):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in self.choices]
        pred = {0: self.choices[0], 1: self.choices[1], 2: self.choices[2], 3: self.choices[3]}[np.argmax(choice_probs)]
        
        return pred
    
    def _correct_answer(self, test_question_df, test_question_number):
        return test_question_df.iloc[test_question_number, 5]

    def _extract_letter(self, text):
        pattern = r'\b([ABCD])\b(?:\s|$|\.)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

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
            writer.writerow(['Prompt Template', self.print_prompt_template()])
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
            "is_correct": pred == correct_answer,
            "prob_A": choice_probs[0],
            "prob_B": choice_probs[1],
            "prob_C": choice_probs[2],
            "prob_D": choice_probs[3]
        }

        # Check if the file exists to determine whether to write headers
        file_exists = os.path.isfile(log_file_path)
        
        with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=row_data.keys())
            
            if not file_exists:
                writer.writeheader()  # Write header if the file is newly created
            
            writer.writerow(row_data)