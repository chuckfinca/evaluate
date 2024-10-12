class DefaultPromptManager():
    def __init__(self, config):
        self.config = config
        self.choices = config['answer_choices']
        self.load_user_prompt_template(config['user_prompt_template'])

    def load_user_prompt_template(self, prompt_template):
        self.prompt_template = prompt_template.get("template", "")
        self.question_template = prompt_template.get("question_template", "")
        self.question_separator = prompt_template.get("question_separator", "\n\n")
        self.instructions_template = prompt_template.get("instructions", "")

    def format_prompt(self, instructions, example_questions_df, test_question_df, test_question_idx):
        example_prompts = []
        for i in range(len(example_questions_df)):
            example_prompts.append(self._format_question(example_questions_df, i, True))

        test_question_prompt = self._format_question(test_question_df, test_question_idx, False)
        
        return self._format_prompt_template(instructions, example_prompts, test_question_prompt)

    def format_instructions(self, subject="{subject}"):
        return self.instructions_template.format(
            subject=subject.replace("high_school_","").replace("college_","").replace("elementary_",""),
            label_a=self.choices[0],
            label_b=self.choices[1],
            label_c=self.choices[2],
            label_d=self.choices[3]
        )

    def print_prompt_template(self):
        example_questions = [f"{{example_{i+1}}}" for i in range(self.config.get('nshot', 0))]
        formatted_instructions = self.format_instructions()
        return self._format_prompt_template(formatted_instructions, example_questions, "{test question}")

    def _format_prompt_template(self, instructions, example_questions, test_question):
        formatted_example_questions = self.question_separator.join(example_questions)
        return self.prompt_template.format(
            instructions=instructions,
            examples=formatted_example_questions,
            question=test_question
        ).strip()

    def _format_question(self, df, row_index, include_answer):
        question, choices, answer = self._process_question_row(df, row_index, include_answer)
        return self._format_question_template(question, choices, answer)

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