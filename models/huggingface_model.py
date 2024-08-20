import os
from aiohttp.web_routedef import static
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel

class HuggingFaceModel(BaseModel):

    def __init__(self, model_name):
        self.model_name = model_name

        self.local_model_path = os.path.join(os.getcwd(), "models", "saved")
        print("-aaa")
        print(self.local_model_path)
        if self._is_model_saved():
            self._load_local_model()
        else:
            self._download_and_save_model()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
    
    def _is_model_saved(self):
        return os.path.exists(self.local_model_path)

    def _load_local_model(self):
        print(f"Loading model from {self.local_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)

    def _download_and_save_model(self):
        print(f"Downloading model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Saving model to {self.local_dir}")
        self.model.save_pretrained(self.local_dir)
        self.tokenizer.save_pretrained(self.local_dir)

    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_answer_probabilities(self, prompt, choices):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        probs = torch.nn.functional.softmax(logits, dim=0)
        return [probs[self.tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in choices]