import dspy
from finca.dspy.signatures.multiple_choice_signature import MMLUSignature

class MultipleChoiceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MMLUSignature)

    def __call__(self, subject, instructions, question, choice_a, choice_b, choice_c, choice_d, answer, **kwargs):
        pred = self.predictor(subject=subject, task_instructions=instructions, question=question, choices_a=choice_a, choice_b=choice_b, choice_c=choice_c, choice_d=choice_d, answer=answer, **kwargs)
        return pred.answer