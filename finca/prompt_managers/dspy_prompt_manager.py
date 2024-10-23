import dspy
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.multiple_choice_prompt_manager import MultipleChoicePromptManager

class MultipleChoiceSignature(dspy.Signature):
    """Defines the signature for multiple choice questions"""
    prompt = dspy.InputField(desc="The context including examples, instructions, and question to answer")
    # question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer must be A, B, C, or D")

class MultipleChoiceProgramRM(dspy.Program):
    """Basic reasoning module for multiple choice questions"""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MultipleChoiceSignature)

    def forward(self, prompt):
        pred = self.predictor(prompt)
        return pred.answer

class DSPyPromptManager(MultipleChoicePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        
    def prepare_prompt(self, subject, examples, question):
        return super().prepare_prompt(subject, examples, question)

    def print_prompt(self) -> str:
        return "DSPyPromptManager using MultipleChoiceProgramRM with ChainOfThought predictor"