import os
from aiohttp.web_routedef import static
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel

class HuggingFaceModel(BaseModel):

    def __init__(self, model_name, project_root, device):
        self.model_name = model_name
        self.token = os.getenv('HF_TOKEN')

        if not self.token:
            raise ValueError("HF_TOKEN not found in .env file at the root of the project")

        self.local_model_path = os.path.join(project_root, 'models', "saved", model_name)

        if self._is_model_saved():
            self._load_local_model()
        else:
            self._download_and_save_model()

        self.device = device
        # self.model.to(self.device)
        self.model.eval()
    
    def _is_model_saved(self):
        return os.path.exists(self.local_model_path)

    def _setup_model(self, model_path):
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            # config=config,
            device_map="auto",
            torch_dtype=torch.float16  # This uses less memory
        )

    def _load_local_model(self):
        print(f"Loading model from {self.local_model_path}")
        self.model = self._setup_model(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)

    def _download_and_save_model(self):
        print(f"Downloading model {self.model_name}")
        self.model = self._setup_model(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Saving model to {self.local_model_path}")
        self.model.save_pretrained(self.local_model_path)
        self.tokenizer.save_pretrained(self.local_model_path)

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