import unittest
from finca.evaluate.orchestrators.mmlu_benchmark_orchestrator import MMLUEvaluationOrchestrator
from finca.prompt_managers.default_prompt_manager import DefaultPromptManager

separator = "\n"

class TestMMLUPromptGeneration(unittest.TestCase):
    
    def setUp(self):
        # Create a dictionary to represent the config
        
        self.config = {
            'answer_choices': ['A', 'B', 'C', 'D'],
            'user_prompt_template': {
                "template": "{instructions}\n{examples}" + separator + "{question}",
                "question_template": "{question}\nA: {choice_a}\nB: {choice_b}\nC: {choice_c}\nD: {choice_d}\nAnswer: {answer}",
                "question_separator": separator,
                "instructions": "Answer the following multiple choice question."
            }
        }
        
        self.prompt_manager = DefaultPromptManager(self.config)

    def test_prompt_generation(self):
        instructions = "These are the instructions."
        example_questions = ["Example 1", "Example 2"]
        test_question = "This is the test question."

        # Case 1: No instructions
        result = self.prompt_manager._format_prompt_template("", example_questions, test_question)
        expected = f"Example 1{separator}Example 2{separator}This is the test question."
        self.assertEqual(result, expected)

        # Case 2: No examples
        result = self.prompt_manager._format_prompt_template(instructions, [], test_question)
        expected = f"These are the instructions.\n{separator}This is the test question."
        self.assertEqual(result, expected)

        # Case 3: No instructions or examples
        result = self.prompt_manager._format_prompt_template("", [], test_question)
        expected = "This is the test question."
        self.assertEqual(result, expected)

        # Case 4: Everything present
        result = self.prompt_manager._format_prompt_template(instructions, example_questions, test_question)
        expected = f"These are the instructions.\nExample 1{separator}Example 2{separator}This is the test question."
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()