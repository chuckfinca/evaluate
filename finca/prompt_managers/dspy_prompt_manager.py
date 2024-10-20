import dspy
import torch
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.task_type import TaskType

class MultipleChoiceExample(dspy.Signature):
    """A single multiple-choice example."""
    question = dspy.InputField()
    choices = dspy.InputField()
    correct_answer = dspy.InputField()

class MultipleChoiceQA(dspy.Signature):
    """Answer a multiple-choice question given instructions and examples."""
    subject = dspy.InputField()
    examples = dspy.InputField(desc="List of MultipleChoiceExample")
    question = dspy.InputField()
    choices = dspy.InputField()
    answer = dspy.OutputField(desc="The letter (A, B, C, or D) of the correct answer")

class MultipleChoiceTeleprompter(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, subject, examples, question, choices):
        instructions = f"Answer the following {subject} question. Choose the best answer from the given options."
        formatted_examples = ""
        for ex in examples:
            formatted_examples += f"Question: {ex.question}\n"
            for letter, choice in zip(self.choices, ex.choices):
                formatted_examples += f"{letter}. {choice}\n"
            formatted_examples += f"Answer: {ex.correct_answer}\n\n"

        formatted_question = f"Question: {question}\n"
        for letter, choice in zip(self.choices, choices):
            formatted_question += f"{letter}. {choice}\n"

        prompt = f"{instructions}\n\n{formatted_examples}Now, please answer this question:\n\n{formatted_question}"
        return prompt

class MultipleChoiceProgram(dspy.Module):
    def __init__(self, subject, examples, question, choices):
        super().__init__()
        self.teleprompter = MultipleChoiceTeleprompter()
        print("aaa")
        self.qa = MultipleChoiceQA(subject, examples, question, choices)
        print("bbb")

    def forward(self, subject, examples, question, choices):
        prompt = self.teleprompter(subject, examples, question, choices)
        return self.qa(subject=subject, examples=examples, question=question, choices=choices, _hint=prompt)


class DSPyPromptManager(BasePromptManager):
    def __init__(self, config, tokenizer=None, model=None):
        super().__init__(config, tokenizer)
        self.choices = ['A', 'B', 'C', 'D']
        self.program = MultipleChoiceProgram()

    def prepare_prompt(self, subject, examples, question):
        dspy_examples = [
            MultipleChoiceExample(
                question=row[0],
                choices=[row[1], row[2], row[3], row[4]],
                correct_answer=row[5]
            ) for _, row in examples.iterrows()
        ]

        result = self.program(
            subject=subject,
            examples=dspy_examples,
            question=question[0],
            choices=[question[1], question[2], question[3], question[4]]
        )

        return result.answer

    def get_expected_output_format(self) -> str:
        return "single_letter"

    def print_prompt(self) -> None:
        print("DSPy Program Structure:")
        print(self.program)

        print("\nExample Prompt (with placeholder data):")
        example_subject = "mathematics"
        example_examples = [
            MultipleChoiceExample(
                question="What is 2 + 2?",
                choices=["3", "4", "5", "6"],
                correct_answer="B"
            )
        ]
        example_question = "What is 3 * 3?"
        example_choices = ["6", "7", "9", "10"]

        example_prompt = self.program.teleprompter(example_subject, example_examples, example_question, example_choices)
        print(example_prompt)