from abc import ABC, abstractmethod

from finca.dspy.dspy_model_wrapper import DSPyModelWrapper

class BaseModelLoader(ABC):
    
    def __init__(self, model, tokenizer):
        self.wrapped_model = DSPyModelWrapper(model, tokenizer)
        self.tokenizer = tokenizer
    