import dspy
from finca.dspy.signatures.multiple_choice_signature import MultipleChoiceSignature

class MultipleChoiceProgram(dspy.Program):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MultipleChoiceSignature)

    def forward(self, **kwargs):
        # Extract prompt from kwargs to ensure it's passed correctly
        prompt = kwargs.get('prompt')
        if prompt is None:
            raise ValueError("Prompt is required for MultipleChoiceProgram")
            
        # Pass prompt as a keyword argument
        pred = self.predictor(prompt=prompt)
        return pred.answer