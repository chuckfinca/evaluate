from abc import ABC, abstractmethod

from finca.model_loaders.dspy_model_wrapper import DSPyModelWrapper

class BaseModelLoader(ABC):
    
    def __init__(self, model, tokenizer, use_dspy=False):
        self.model = DSPyModelWrapper(model, tokenizer, use_dspy)
        self.tokenizer = tokenizer