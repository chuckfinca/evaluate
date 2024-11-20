import dspy
from finca.dspy.signatures.multiple_choice_signature import MMLUSignature

class MultipleChoiceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MMLUSignature)

    def __call__(self, prompt_object, **kwargs):
        pred = self.predictor(subject=prompt_object.subject, task_instructions=prompt_object.instructions, question=prompt_object.question, choice_a=prompt_object.choices[0], choice_b=prompt_object.choices[1], choice_c=prompt_object.choices[2], choice_d=prompt_object.choices[3], answer=prompt_object.answer, **kwargs)
        return pred.answer