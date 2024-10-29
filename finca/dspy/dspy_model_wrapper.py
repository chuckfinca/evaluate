import dspy
from finca.dspy.dspy_lm import DSPyLM
from finca.dspy.program_registry import DSPyProgramRegistry
from finca.dspy.signatures.multiple_choice_signature import MMLUSignature
from finca.dspy.adapters.mmlu_adapter import MMLUAdapter

class DSPyModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.program_registry = DSPyProgramRegistry()
        self.dspy_initialized = False
    
    def setup_dspy_environment(self):
        """Lazy initialization of DSPy environment when first program is registered"""
        if not self.dspy_initialized:
            kwargs = {
                'temperature': 0.0,
                'max_tokens': 100,
                'stop': None,
                'n': 1
            }
            self.dspy_lm = DSPyLM(self.model, self.tokenizer, **kwargs)
            dspy.configure(lm=self.dspy_lm) #, adapter=MMLUAdapter())
            self.dspy_initialized = True
    
    def register_program(self, program_name: str):
        """Register a DSPy program"""
        self.setup_dspy_environment()
        self.program_registry.register(program_name)

    @property
    def has_dspy_programs(self) -> bool:
        """Check if any DSPy programs are registered"""
        return len(self.program_registry._programs) > 0
    
    def __call__(self, prompt_object, **kwargs):
        # Define the predictor.
        predictor = dspy.Predict(MMLUSignature)
        
        mo = prompt_object

        # Call the predictor on a particular input.
        pred = predictor(subject=mo.subject, task_instructions=mo.instructions, question=mo.question, choice_a=mo.choices[0], choice_b=mo.choices[1], choice_c=mo.choices[2], choice_d=mo.choices[3], answer=mo.answer, **kwargs)
        return pred
        # Handle DSPy program execution
        if "program_name" in kwargs:
            program_name = kwargs.pop("program_name")
            try:
                program = self.program_registry.get_program(program_name)
                return program(prompt, **kwargs)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error executing DSPy program {program_name}: {str(e)}")
        
        # Handle regular model inference
        elif prompt:
            try:
                if kwargs.pop("generate", False):
                    return self.model.generate(prompt, **kwargs)
                return self.model(prompt, **kwargs)
            except Exception as e:
                raise ValueError(f"Error during model inference: {str(e)}")
        
        raise ValueError("Missing prompt or program_name in arguments")