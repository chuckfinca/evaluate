import dspy
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.default_prompt_manager import DefaultPromptManager
from finca.prompt_managers.task_type import TaskType

class DSPyPromptManager(BasePromptManager):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.lm = config.get('lm')  # This should be a DSPy language model

    def prepare_prompt(self, subject, examples, question):
        if self.task_type == TaskType.MULTIPLE_CHOICE:
            return self._prepare_multiple_choice_prompt(subject, examples, question)
        elif self.task_type == TaskType.OPEN_ENDED:
            return self._prepare_open_ended_prompt(subject, examples, question)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _prepare_multiple_choice_prompt(self, subject, examples, question):
        import dspy

        class MultipleChoicePrompt(dspy.Signature):
            """Answer multiple-choice questions."""
            subject = dspy.InputField()
            examples = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField(desc="The letter (A, B, C, or D) corresponding to the correct answer.")

        formatted_examples = self._format_examples(examples)
        formatted_question = self._format_question(question)

        prompter = dspy.Predict(MultipleChoicePrompt)
        return prompter(subject=subject.replace("_", " "), examples=formatted_examples, question=formatted_question)

    def _prepare_open_ended_prompt(self, subject, examples, question):
        import dspy

        class OpenEndedPrompt(dspy.Signature):
            """Answer open-ended questions based on the given context."""
            subject = dspy.InputField()
            examples = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField(desc="A detailed answer to the question based on the context.")

        formatted_examples = self._format_examples(examples)
        formatted_question = self._format_question(question)

        prompter = dspy.Predict(OpenEndedPrompt)
        return prompter(subject=subject.replace("_", " "), examples=formatted_examples, question=formatted_question)

    def get_expected_output_format(self) -> str:
        return "dspy_format"

    def print_prompt(self) -> None:
        print("DSPy Prompt Manager does not have a static prompt template.")
        print("Prompts are dynamically generated based on the task type and inputs.")

    # Reuse the _format_examples and _format_question methods from DefaultPromptManager
    _format_examples = DefaultPromptManager._format_examples
    _format_question = DefaultPromptManager._format_question