from abc import ABC, abstractmethod
from finca.prompt_managers.task_type import TaskType

class BasePromptManager(ABC):
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def prepare_prompt(self, subject, examples, question):
        """Prepare a PromptConfig for a given task type and parameters."""
        pass

    @abstractmethod
    def get_expected_output_format(self) -> str:
        """Return the expected format of the model's output for a given task type."""
        pass

    @abstractmethod
    def print_prompt(self) -> None:
        """Print the formatted prompt template."""
        pass

    def apply_chat_template(self, messages):
        if self.use_chat_template and self.tokenizer:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        elif isinstance(messages, list):
            return messages[1]['content']  # Use the user message if not using chat_template
        else:
            return messages  # Return as is if it's already a string
