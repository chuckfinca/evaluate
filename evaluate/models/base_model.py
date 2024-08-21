from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate_answer(self, prompt):
        pass

    @abstractmethod
    def get_answer_probabilities(self, prompt, choices):
        pass

    