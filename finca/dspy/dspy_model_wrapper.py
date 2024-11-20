import dspy
from finca.dspy.dspy_lm import DSPyLM
from finca.dspy.program_registry import DSPyProgramRegistry
from finca.logs.logger import logger

class DSPyModelWrapper:
    def __init__(self, model_name):
        self.dspy_lm = dspy.HFModel(model_name)
        # self.dspy_lm = DSPyLM(model, tokenizer)
        dspy.configure(lm=self.dspy_lm)

        self.program_registry = DSPyProgramRegistry()
        self.log_prompt = False
    
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
                output = program(prompt_object, **kwargs)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error executing DSPy program {program_name}: {str(e)}")
        
        # Handle regular model inference
        elif prompt_object:
            try:
                if kwargs.pop("generate", False):
                    output = self.model.generate(prompt_object, **kwargs)
                return self.model(prompt_object, **kwargs)
            except Exception as e:
                raise ValueError(f"Error during model inference: {str(e)}")
        
        if self.log_prompt:
            if "program_name" in kwargs:
                prompt = dspy.inspect_history(n=1)
                
            elif prompt_object:
                prompt = prompt_object
            logger.log.info(prompt)
        if output is not None:
          return output

        raise ValueError("Missing prompt or program_name in arguments")