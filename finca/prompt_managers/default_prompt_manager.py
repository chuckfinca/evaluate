import re

from finca.prompt_managers.multiple_choice_prompt_manager import MultipleChoicePromptManager

class DefaultPromptManager(MultipleChoicePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.use_chat_template = config.get('use_chat_template', False)
        self.system_prompt = config.get('system_prompt', "")

    def prepare_prompt(self, subject, examples, question):
        instructions, examples, question = super().prepare_prompt(subject, examples, question)

        prompt = self.prompt_template.format(
            instructions=instructions,
            examples=examples,
            question=question
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

    def print_prompt(self) -> None:
        example_questions = [f"{{example_{i+1}}}" for i in range(5)]  # Assuming max 5 example questions
        formatted_instructions = self.format_instructions("{subject}")
        print(self._format_prompt_template(formatted_instructions, example_questions, "{test question}"))

    def _extract_template_variables(self) -> list:
        return re.findall(r'\{(\w+)\}', self.prompt_template)

    def _format_prompt_template(self, instructions, example_questions, test_question):
        formatted_example_questions = self.question_separator.join(example_questions)
        return self.prompt_template.format(
            instructions=instructions,
            examples=formatted_example_questions,
            question=test_question
        ).strip()