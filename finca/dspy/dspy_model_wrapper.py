import dspy
from finca.dspy.dspy_lm import DSPyLM
from finca.dspy.program_registry import DSPyProgramRegistry

class DSPyModelWrapper:
    def __init__(self, model, tokenizer):
        self.dspy_lm = DSPyLM(model, tokenizer)
        dspy.configure(lm=self.dspy_lm)

        self.device = model.device
        self.program_registry = DSPyProgramRegistry()
    
    def register_program(self, program_name: str):
        """Register a DSPy program"""
        self.program_registry.register(program_name)

    @property
    def has_dspy_programs(self) -> bool:
        """Check if any DSPy programs are registered"""
        return len(self.program_registry._programs) > 0
    
    def __call__(self, prompt_object, **kwargs):

        # Handle DSPy program execution
        if "program_name" in kwargs:
            program_name = kwargs.pop("program_name")
            try:
                program = self.program_registry.get_program(program_name)
                pred = program(prompt_object, **kwargs)
                return pred.answer
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error executing DSPy program {program_name}: {str(e)}")
        
        # Handle regular model inference
        elif prompt_object:
            try:
                if kwargs.pop("generate", False):
                    return self.model.generate(prompt_object, **kwargs)
                return self.model(prompt_object, **kwargs)
            except Exception as e:
                raise ValueError(f"Error during model inference: {str(e)}")
        
        raise ValueError("Missing prompt or program_name in arguments")