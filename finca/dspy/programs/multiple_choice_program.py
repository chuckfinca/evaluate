import dspy

from finca.dspy.signatures.multiple_choice_signature import MultipleChoiceSignature

class MultipleChoiceProgram(dspy.Program):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MultipleChoiceSignature)

    def forward(self, prompt):
        pred = self.predictor(prompt=prompt)
        return pred.answer