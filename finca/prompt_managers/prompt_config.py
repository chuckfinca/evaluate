import dspy

class PromptConfig:
    def __init__(self, template, fields, metadata = None):
        self.template = template
        self.fields = fields
        self.metadata = metadata or {}

    def format_prompt(self) -> str:
        """Format the prompt using the template and fields."""
        if isinstance(self.template, str):
            return self.template.format(**self.fields)
        elif isinstance(self.template, dspy.Signature):
            return str(self.template)
        else:
            raise ValueError("Unsupported template type")