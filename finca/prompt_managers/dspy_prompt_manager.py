import dspy
from finca.prompt_managers.base_prompt_manager import BasePromptManager
from finca.prompt_managers.prompt_config import PromptConfig
from finca.prompt_managers.task_type import TaskType

class DSPyPromptManager(BasePromptManager):
    def __init__(self, config):
        super().__init__(config)
        
        self.lm = dspy.LM(config['model_name'])

    def prepare_prompt_config(self, **kwargs) -> PromptConfig:
        if self.task_type == TaskType.MULTIPLE_CHOICE:
            return self._prepare_multiple_choice_config(**kwargs)
        elif self.task_type == TaskType.OPEN_ENDED:
            return self._prepare_open_ended_config(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def get_expected_output_format(self) -> str:
        return "dspy_format"

    def _prepare_multiple_choice_config(self, subject, examples, question) -> PromptConfig:
        class MultipleChoicePrompt(dspy.Signature):
            """Answer multiple-choice questions."""
            subject = dspy.InputField()
            examples = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField(desc="The letter (A, B, C, or D) corresponding to the correct answer.")

        prompter = dspy.Predict(MultipleChoicePrompt)
        return PromptConfig(
            template=str(prompter.signature),
            fields={"subject": subject, "examples": examples, "question": question},
            metadata={"prompter": prompter}
        )

    def _prepare_open_ended_config(self, context, question) -> PromptConfig:
        class OpenEndedPrompt(dspy.Signature):
            """Answer open-ended questions based on the given context."""
            context = dspy.InputField()
            question = dspy.InputField()
            answer = dspy.OutputField(desc="A detailed answer to the question based on the context.")

        prompter = dspy.Predict(OpenEndedPrompt)
        return PromptConfig(
            template=str(prompter.signature),
            fields={"context": context, "question": question},
            metadata={"prompter": prompter}
        )
