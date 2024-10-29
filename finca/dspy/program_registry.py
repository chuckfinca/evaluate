import dspy
from finca.dspy.programs.multiple_choice_program import MultipleChoiceModule

class DSPyProgramRegistry:
    """Registry for DSPy programs"""
    
    PROGRAM_CLASSES = {
        "MultipleChoiceProgram": MultipleChoiceModule
    }
    
    def __init__(self):
        self._programs = {}
        self._default_program = None
    
    def register(self, program_name: str):
        """Register a DSPy program"""
        if program_name not in self.PROGRAM_CLASSES:
            raise ValueError(f"Unknown program: {program_name}")
            
        program_class = self.PROGRAM_CLASSES[program_name]
        self._programs[program_name] = program_class
    
    def get_program(self, name: str = None) -> dspy.Program:
        """Get a program instance by name"""
        if name is None:
            if self._default_program is None:
                raise ValueError("No default program set and no program name provided")
            return self._default_program()
            
        if name not in self._programs:
            raise KeyError(f"No DSPy program registered with name: {name}")
        
        return self._programs[name]()