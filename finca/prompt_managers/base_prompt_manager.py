from abc import ABC, abstractmethod

class BasePromptManager(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def format_prompt(self, instructions, example_questions_df, test_question_df, test_question_idx):
        pass

    @abstractmethod
    def format_instructions(self, subject):
        pass

    @abstractmethod
    def print_prompt_template(self):
        pass