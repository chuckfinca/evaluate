import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from finca.logs.logger import logger
from finca.model_loaders.model_loader_config import get_model_loader_config
from finca.utils.path_utils import path_to_package_data
from finca.model_loaders.base_model_loader import BaseModelLoader

class HuggingFaceModelLoader(BaseModelLoader):

    def __init__(self, model_name):
        self.model_name = model_name
        self.token = os.getenv('HF_TOKEN')

        if not self.token:
            raise ValueError("HF_TOKEN not found in .env file at the root of the project")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.float16

        package_data_directory = path_to_package_data()
        self.local_model_path = os.path.join(package_data_directory, 'models', model_name)

        if self._is_model_saved():
            model, tokenizer = self._load_local_model()
        else:
            model, tokenizer = self._download_model()

        model.to(self.device).to(self.dtype)
        
        # Call the base class initializer with the loaded model and tokenizer
        super().__init__(model, tokenizer)
        
        logger.log.info(f"device: {self.device}")
        logger.log.info(f"dtype: {self.dtype}")
        logger.log.info(f"model: {type(self.model).__name__}")
        logger.log.info(f"tokenizer: {type(self.tokenizer).__name__}")
        
        logger.log.info("Special Tokens:")
        logger.log.info(self.tokenizer.sep_token)
        logger.log.info(self.tokenizer.eos_token)
        logger.log.info(self.tokenizer.all_special_tokens)
        logger.log.debug(self.tokenizer)
    
    def _is_model_saved(self):
        return os.path.exists(self.local_model_path)

    def _setup_model(self, model_path):
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        )

    def _load_local_model(self):
        logger.log.info(f"Loading model from {self.local_model_path}")
        model = self._setup_model(self.local_model_path)
        tokenizer = self._setup_tokenizer(self.model_name)
        return model, tokenizer

    def _download_model(self):
        logger.log.info(f"Downloading model {self.model_name}")
        model = self._setup_model(self.model_name)
        tokenizer = self._setup_tokenizer(self.model_name)
        return model, tokenizer
        
    def _save_model(self):
        if not self._is_model_saved():
            logger.log.info(f"Starting to save model to {self.local_model_path}")
            try:
                self.model.save_pretrained(self.local_model_path)
                self.tokenizer.save_pretrained(self.local_model_path)
                logger.log.info(f"Model saved to {self.local_model_path}")
            except Exception as e:
                logger.log.error(f"Error saving model: {str(e)}")
        
    def _setup_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if config := get_model_loader_config(model_name):
            tokenizer.bos_id: int = config["bos_id"]
            tokenizer.eos_id: int = config["eos_id"]
            tokenizer.pad_id: int = config["pad_id"]
            tokenizer.stop_tokens = config["stop_tokens"]
        return tokenizer
        
