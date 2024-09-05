import unittest
from evaluate.orchestrators.mmlu_benchmark_orchestrator import MMLUEvaluationOrchestrator

class TestMMLUPromptGeneration(unittest.TestCase):
    def setUp(self):
        # Create a minimal MMLUEvaluationOrchestrator instance
        self.orchestrator = MMLUEvaluationOrchestrator(None, None, 'mmlu', None, None)

    def test_prompt_generation(self):
        instructions = "These are the instructions."
        example_questions = ["Example 1", "Example 2"]
        test_question = "This is the test question."

        # Case 1: No instructions
        result = self.orchestrator._format_prompt_template("", example_questions, test_question)
        expected = "Example 1\nExample 2\nThis is the test question."
        self.assertEqual(result, expected)

        # Case 2: No examples
        result = self.orchestrator._format_prompt_template(instructions, [], test_question)
        expected = "These are the instructions.\nThis is the test question."
        self.assertEqual(result, expected)

        # Case 3: No instructions or examples
        result = self.orchestrator._format_prompt_template("", [], test_question)
        expected = "This is the test question."
        self.assertEqual(result, expected)

        # Case 4: Everything present
        result = self.orchestrator._format_prompt_template(instructions, example_questions, test_question)
        expected = "These are the instructions.\nExample 1\nExample 2\nThis is the test question."
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()