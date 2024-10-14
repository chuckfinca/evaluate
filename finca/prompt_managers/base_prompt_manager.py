from abc import ABC, abstractmethod
from finca.prompt_managers.prompt_config import PromptConfig
from finca.prompt_managers.task_type import TaskType

class BasePromptManager(ABC):
    def __init__(self, config):
        self.config = config
        self.task_type = TaskType(self.config["task_type"])

    @abstractmethod
    def prepare_prompt_config(self, **kwargs) -> PromptConfig:
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