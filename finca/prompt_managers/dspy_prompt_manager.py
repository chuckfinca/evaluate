import dspy
from finca.prompt_managers.base_prompt_manager import BasePromptManager

class DSPyPromptManager(BasePromptManager):
    def __init__(self, config):
        super().__init__(config)
        self.choices = config['answer_choices']
        self.lm = dspy.LM(config['model_name'])
        self.prompt_template = dspy.Template(config['dspy_prompt_template'])

    def format_prompt(self, instructions, example_questions_df, test_question_df, test_question_idx):
        examples = [self._format_question(example_questions_df, i, True) for i in range(len(example_questions_df))]
        test_question = self._format_question(test_question_df, test_question_idx, False)
        return self.prompt_template(instructions=instructions, examples=examples, question=test_question)

    def format_instructions(self, subject):
        return self.config.get('instructions_template', '').format(subject=subject, **{f'label_{c.lower()}': c for c in self.choices})

    def print_prompt_template(self):
        return str(self.prompt_template)

    def _format_question(self, df, row_index, include_answer):
        row = df.iloc[row_index]
        question = row[0]
        choices = {self.choices[i]: row[i+1] for i in range(4)}
        answer = row[5] if include_answer else None
        return dspy.Example(question=question, choices=choices, answer=answer)