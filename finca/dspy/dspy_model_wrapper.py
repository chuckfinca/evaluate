import dspy

from finca.dspy.dspy_lm import DSPyLM
from finca.dspy.program_registry import DSPyProgramRegistry

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
            dspy.settings.configure(lm=self.dspy_lm)
            self.dspy_initialized = True
    
    def register_program(self, name: str, program_class: str):
        """Register a DSPy program and ensure environment is setup"""
        self.setup_dspy_environment()
        self.program_registry.register(name, program_class)
    
    @property
    def has_dspy_programs(self) -> bool:
        """Check if any DSPy programs are registered"""
        return len(self.program_registry._programs) > 0
    
    def __call__(self, prompt, program_name=None, **kwargs):
        if self.has_dspy_programs:
            try:
                program = self.program_registry.get_program(program_name)
                return program(prompt=prompt, **kwargs)
            except (KeyError, ValueError):
                # Fallback to basic LM if program not found or no default set
                return self.dspy_lm(prompt, **kwargs)
        else:
            if kwargs.pop("generate", False):
                return self._default_generate(prompt, **kwargs)
            return self._default(prompt, **kwargs)
