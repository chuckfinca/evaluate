import torch
import dspy
from finca.logs.logger import logger

class CustomLM(dspy.LM):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs # required dspy attribute

    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class DSPyModelWrapper:
    def __init__(self, model, tokenizer, use_dspy=False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_dspy = use_dspy
        self.device = model.device
        if use_dspy:
            self.setup_dspy_environment(model, tokenizer)

    def setup_dspy_environment(self, model, tokenizer):
        kwargs = {
            'temperature': 0.0,  # deterministic
            'max_tokens': 100,   # reasonable default
            'stop': None,        # no default stop sequences
            'n': 1             # single completion
        }
        self.dspy_lm = CustomLM(model, tokenizer, **kwargs)
        dspy.settings.configure(lm=self.dspy_lm)

    def __call__(self, prompt, **kwargs):
        if self.use_dspy:
            here the prompt manager needs to run the program instead of just calling the model
            return self.dspy_lm(prompt, **kwargs)
        else:
            if "generate" in kwargs and kwargs["generate"] == True:
                kwargs.pop('generate', None)
                return self._default_generate(prompt, **kwargs)
            else:
                return self._default(prompt, **kwargs)

    def generate(self, prompt, **kwargs):
        if self.use_dspy:
            return self.dspy_lm(prompt, **kwargs)
        else:
            return self._default_generate(prompt, **kwargs)
    
    def _default(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs

    def _default_generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)