import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate.utils.path_utils import get_user_directory

class HuggingFaceModel():

    def __init__(self, model_name):
        self.model_name = model_name
        self.token = os.getenv('HF_TOKEN')

        if not self.token:
            raise ValueError("HF_TOKEN not found in .env file at the root of the project")

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Determine the appropriate dtype
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.float16
        print(f"Using dtype: {self.dtype}")

        user_directory = get_user_directory()
        self.local_model_path = os.path.join(user_directory, 'models', model_name)

        if self._is_model_saved():
            self._load_local_model()
        else:
            self._download_and_save_model()

        self.model.to(self.device).to(self.dtype)
    
    def _is_model_saved(self):
        return os.path.exists(self.local_model_path)

    def _setup_model(self, model_path):
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            # low_cpu_mem_usage=True,
            torch_dtype=self.dtype
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