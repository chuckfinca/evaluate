from abc import ABC, abstractmethod

from finca.dspy.dspy_model_wrapper import DSPyModelWrapper

class BaseModelLoader(ABC):
    
    def __init__(self, model_name):
        self.wrapped_model = DSPyModelWrapper(model_name)
    