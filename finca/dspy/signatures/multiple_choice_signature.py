import dspy

class MultipleChoiceSignature(dspy.Signature):
    """Defines the signature for multiple choice questions"""
    prompt = dspy.InputField(desc="The context including examples, instructions, and question to answer")
    # question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer must be A, B, C, or D")