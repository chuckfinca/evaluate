
import re
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.task_type import TaskType

class DefaultPromptManager(BasePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.user_prompt_template = config.get('user_prompt_template', {})
        self.prompt_template = self.user_prompt_template.get('template', "{instructions}\n{question}")
        self.question_template = self.user_prompt_template.get('question_template', "{question} Answer Choices: ({label_a}){choice_a} ({label_b}){choice_b} ({label_c}){choice_c} ({label_d}){choice_d}\nA: Among A through E, the answer is")
        self.question_separator = self.user_prompt_template.get('question_separator', "\n\n")
        self.instructions = self.user_prompt_template.get('instructions', "Give your answer in the format \"The answer is therefore <{label_a}, {label_b}, {label_c}, {label_d}>\". Failure to comply with the answer formatting will result in no credit.")
        self.choices = config.get('answer_choices', ['A', 'B', 'C', 'D'])
        self.system_prompt = config.get('system_prompt', "")
        # self.prompt_template = config.get('prompt_template', "{subject}\n\nExamples:\n{examples}\n\nQuestion: {question}\nAnswer:")
        self.choices = config.get('answer_choices', ['A', 'B', 'C', 'D'])

    def prepare_prompt(self, subject, examples, question):
        formatted_examples = self._format_examples(examples)
        formatted_question = self._format_question(question)
        
        prompt = self.prompt_template.format(
            subject=subject.replace("_", " "),
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
        print("Prompt Template:")
        print(self.prompt_template)
        print("\nTemplate Variables:")
        for variable in self._extract_template_variables():
            print(f"- {variable}")

    def _extract_template_variables(self) -> list:
        return re.findall(r'\{(\w+)\}', self.prompt_template)

    def _format_examples(self, examples):
        formatted_examples = []
        for _, example in examples.iterrows():
            formatted_example = self._format_question(example)
            formatted_examples.append(formatted_example)
        return "\n\n".join(formatted_examples)

    def _format_question(self, question):
        formatted_question = f"Q: {question[0]}\n"
        items = question.items()
        for i in range(len(self.choices)):
            formatted_question += f"{self.choices[i]}) {question[i+1]}\n"
        if len(question) == 5:
            formatted_question += f"A: {question[5]}"
        return formatted_question.strip()

    def apply_chat_template(self, messages):
        if self.use_chat_template and self.tokenizer:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        elif isinstance(messages, list):
            return messages[1]['content']  # Use the user message if not using chat_template
        else:
            return messages  # Return as is if it's already a string