import dspy

# Thanks to https://github.com/epec254/dspy_examples/blob/main/dpsy_mmlu.py
class MMLUSignature(dspy.Signature):
    # what is in the block quote below is included in the content key in the SYSTEM message dictionary (i.e. is part of the SYSTEM prompt)
    """You answer questions. At the end of the question you always give an answer and nothing else. You must pick an answer. You always give only one answer and that one answer is the one you think is best. You always give the answer in the form of the answer choice letter."""

    subject = dspy.InputField(desc="Subject")
    task_instructions = dspy.InputField(desc="Instructions")
    question = dspy.InputField(desc="Question")
    choice_a = dspy.InputField(desc="First multiple choice answer")
    choice_b = dspy.InputField(desc="Second multiple choice answer")
    choice_c = dspy.InputField(desc="Third multiple choice answer")
    choice_d = dspy.InputField(desc="Fourth multiple choice answer")
    answer = dspy.OutputField(desc="The correct answer's multiple choice label")