from abc import abstractmethod
from finca.prompt_managers.base_prompt_manager import BasePromptManager

class MultipleChoicePromptManager(BasePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        user_prompt_template = config.get('user_prompt_template', {})
        self.prompt_template = user_prompt_template.get('template', "Missing 'template' in the config")
        self.question_template = user_prompt_template.get('question_template', "Missing 'question_template' in the config")
        self.question_separator = user_prompt_template.get('question_separator', "Missing 'question_separator' in the config")
        self.instructions_template = user_prompt_template.get('instructions', "Missing 'instructions' in the config")
        self.choices = config.get('answer_choices', ['A', 'B', 'C', 'D'])
        
    def prepare_prompt(self, subject, examples, question):
        formatted_instructions = self.format_instructions(subject)
        formatted_examples = self._format_examples(examples)
        formatted_question = self._format_question(question, False)

        return formatted_instructions, formatted_examples, formatted_question

    def print_prompt(self) -> None:
        print("woot!")
        example_questions = [f"{{example_{i+1}}}" for i in range(5)]  # Assuming max 5 example questions
        formatted_instructions = self.format_instructions("{subject}")
        # print(self._format_prompt_template(formatted_instructions, example_questions, "{test question}"))

    def format_instructions(self, subject):
        return self.instructions_template.format(
            subject=subject,
            label_a=self.choices[0],
            label_b=self.choices[1],
            label_c=self.choices[2],
            label_d=self.choices[3]
        )

    def _format_examples(self, examples):
        formatted_examples = []
        for _, example in examples.iterrows():
            formatted_example = self._format_question(example, True)
            formatted_examples.append(formatted_example)
        return self.question_separator.join(formatted_examples)

    def _format_question(self, question, include_answer):
        question_text = question[0]
        choices = {
            self.choices[0]: question[1],
            self.choices[1]: question[2],
            self.choices[2]: question[3],
            self.choices[3]: question[4]
        }
        answer = question[5] if include_answer else None

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