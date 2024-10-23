import importlib
import dspy

def resolve_program_class(class_path: str) -> dspy.Program:
    """
    Converts a string class path to an actual class object.
    
    Args:
        class_path (str): Full path to the class (e.g., "finca.dspy.programs.multiple_choice_program.MultipleChoiceProgram")
        
    Returns:
        Type[dspy.Program]: The actual class object
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class cannot be found in the module
    """
    try:
        # Split the path into module path and class name
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class from the module
        program_class = getattr(module, class_name)
        
        # Verify it's a DSPy program
        if not issubclass(program_class, dspy.Program):
            raise TypeError(f"Class {class_name} is not a subclass of dspy.Program")
            
        return program_class
    
    except ImportError as e:
        raise ImportError(f"Could not import module for class {class_path}: {str(e)}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {class_name} in module {module_path}: {str(e)}")

class DSPyProgramRegistry:
    """Registry for DSPy programs"""
    
    def __init__(self):
        self._programs = {}
        self._default_program = None
    
    def register(self, name: str, program_class: dspy.Program):
        """
        Register a DSPy program class.
        
        Args:
            name (str): Name to register the program under
            program_class (dspy.Program): The program class to register
        """
        if not issubclass(program_class, dspy.Program):
            raise TypeError(f"Class {program_class.__name__} is not a subclass of dspy.Program")
            
        self._programs[name] = program_class
    
    def set_default(self, program_class: dspy.Program):
        """Set the default program class"""
        if not issubclass(program_class, dspy.Program):
            raise TypeError(f"Class {program_class.__name__} is not a subclass of dspy.Program")
            
        self._default_program = program_class
    
    def get_program(self, name: str = None) -> dspy.Program:
        """
        Get a program instance by name or the default if none specified.
        
        Args:
            name (str, optional): Name of the registered program to get
            
        Returns:
            dspy.Program: Instance of the requested program
        
        Raises:
            ValueError: If no default program is set and no name provided
            KeyError: If the requested program name is not registered
        """
        if name is None:
            if self._default_program is None:
                raise ValueError("No default program set and no program name provided")
            return self._default_program()
            
        if name not in self._programs:
            raise KeyError(f"No DSPy program registered with name: {name}")
        
        return self._programs[name]()
    
    def get_program_class(self, name: str) -> dspy.Program:
        """Get the program class (not instance) by name"""
        if name not in self._programs:
            raise KeyError(f"No DSPy program registered with name: {name}")
        return self._programs[name]