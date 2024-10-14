from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.prompt_config import PromptConfig
from finca.prompt_managers.task_type import TaskType

class DefaultPromptManager(BasePromptManager):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_template = config.get('prompt_template', "{question}\n{answer}")

    def prepare_prompt_config(self, **kwargs) -> PromptConfig:
        if self.task_type == TaskType.MultipleChoice:
            return self._prepare_multiple_choice_config(**kwargs)
        elif self.task_type == TaskType.OpenEnded:
            return self._prepare_open_ended_config(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def get_expected_output_format(self) -> str:
        if self.task_type == TaskType.MultipleChoice:
            return "single_letter"
        elif self.task_type == TaskType.OpenEnded:
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
        """Extract variable names from the prompt template."""
        import re
        return re.findall(r'\{(\w+)\}', self.prompt_template)

    def _prepare_multiple_choice_config(self, subject: str, examples: list, question: str) -> PromptConfig:
        fields = {
            "subject": subject,
            "examples": self._format_examples(examples),
            "question": question,
            "choices": ", ".join(self.config['answer_choices'])
        }
        return PromptConfig(self.prompt_template, fields)

    def _prepare_open_ended_config(self, context: str, question: str) -> PromptConfig:
        fields = {
            "context": context,
            "question": question
        }
        return PromptConfig(self.prompt_template, fields)

    def _format_examples(self, examples: list) -> str:
        return "\n\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in examples])
