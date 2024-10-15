
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.default_prompt_manager import DefaultPromptManager
from finca.prompt_managers.dspy_prompt_manager import DSPyPromptManager

class PromptManagerFactory:
    @staticmethod
    def create(config, tokenizer=None):
        prompt_manager_type = config.get('prompt_manager', 'default')
        if prompt_manager_type == 'default':
            return DefaultPromptManager(config, tokenizer)
        elif prompt_manager_type == 'dspy':
            return DSPyPromptManager(config, tokenizer)
        else:
            raise ValueError(f"Unsupported prompt manager type: {prompt_manager_type}")