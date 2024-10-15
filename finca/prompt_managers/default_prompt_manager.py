import re
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.task_type import TaskType

class DefaultPromptManager(BasePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.user_prompt_template = config.get('user_prompt_template', {})
        self.prompt_template = self.user_prompt_template.get('template', "Missing 'template' in the config")
        self.question_template = self.user_prompt_template.get('question_template', "Missing 'question_template' in the config")
        self.question_separator = self.user_prompt_template.get('question_separator', "Missing 'question_separator' in the config")
        self.instructions_template = self.user_prompt_template.get('instructions', "Missing 'instructions' in the config")
        self.choices = config.get('answer_choices', ['A', 'B', 'C', 'D'])
        self.system_prompt = config.get('system_prompt', "")
        self.use_chat_template = config.get('use_chat_template', False)
        self.task_type = config.get('task_type', TaskType.MULTIPLE_CHOICE)

    def prepare_prompt(self, subject, examples, question):
        formatted_instructions = self.format_instructions(subject)
        formatted_examples = self._format_examples(examples)
        formatted_question = self._format_question(question)

        prompt = self.prompt_template.format(
            instructions=formatted_instructions,
            examples=formatted_examples,
            question=formatted_question
        )

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
            return self.apply_chat_template(messages)
        else:
            return prompt

    def get_expected_output_format(self) -> str:
        if self.task_type == TaskType.MULTIPLE_CHOICE:
            return "single_letter"
        elif self.task_type == TaskType.OPEN_ENDED:
            return "free_text"
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def print_prompt(self) -> None:
        example_questions = [f"{{example_{i+1}}}" for i in range(5)]  # Assuming max 5 example questions
        formatted_instructions = self.format_instructions("{subject}")
        print(self._format_prompt_template(formatted_instructions, example_questions, "{test question}"))

    def _extract_template_variables(self) -> list:
        return re.findall(r'\{(\w+)\}', self.prompt_template)

    def _format_examples(self, examples):
        formatted_examples = []
        for _, example in examples.iterrows():
            formatted_example = self._format_question(example)
            formatted_examples.append(formatted_example)
        return self.question_separator.join(formatted_examples)

    def _format_question(self, question):
        question_text = question[0]
        choices = {
            self.choices[0]: question[1],
            self.choices[1]: question[2],
            self.choices[2]: question[3],
            self.choices[3]: question[4]
        }
        answer = question[5] if len(question) == 5 else None

        return self._format_question_template(question_text, choices, answer)

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

    def _format_prompt_template(self, instructions, example_questions, test_question):
        formatted_example_questions = self.question_separator.join(example_questions)
        return self.prompt_template.format(
            instructions=instructions,
            examples=formatted_example_questions,
            question=test_question
        ).strip()

    def format_instructions(self, subject):
        return self.instructions_template.format(
            subject=subject,
            label_a=self.choices[0],
            label_b=self.choices[1],
            label_c=self.choices[2],
            label_d=self.choices[3]
        )