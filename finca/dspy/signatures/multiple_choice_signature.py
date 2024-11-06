import dspy

# Thanks to https://github.com/epec254/dspy_examples/blob/main/dpsy_mmlu.py
class MMLUSignature(dspy.Signature):
    # what is in the block quote below is included in the content key in the SYSTEM message dictionary (i.e. is part of the SYSTEM prompt)
    """Solve tricky multiple choice problems about various subjects. There are 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem solving ability. Subjects range from traditional areas, such as mathematics and history, to more specialized areas like law and ethics. Some require you to answer a question, some require you to fill in the blank, some require you to finish the question with the correct answer."""

    subject = dspy.InputField(desc="the subject of the question")
    task_instructions = dspy.InputField(desc="the instructions")
    question = dspy.InputField(desc="the question to be answered with one of the choices")
    choice_a = dspy.InputField(desc="the first choice you can select from")
    choice_b = dspy.InputField(desc="the second choice you can select from")
    choice_c = dspy.InputField(desc="the third choice you can select from")
    choice_d = dspy.InputField(desc="the fourth choice you can select from")
    answer = dspy.OutputField(desc="The answer which is always one choice_a, choice_b, choice_c, or choice_d - NOT the answer itself")