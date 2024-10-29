import torch
import dspy

class DSPyLM(dspy.LM):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs # required dspy attribute

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            chat_template_supported = self._check_chat_template(messages)
            if chat_template_supported:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = prompt or "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)

        # Create tensors directly on the target device
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            device=self.model.device  # Tensors created directly on GPU if model is on GPU
        )
        
        with torch.no_grad():
            generation_kwargs = {
                'max_new_tokens': len(prompt) + 100,
                'pad_token_id': self.tokenizer.eos_token_id,
                'do_sample': False,
                **kwargs
            }
            output = self.model.generate(**inputs, **generation_kwargs)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _check_chat_template(self, messages):
        try:
            for message in messages:
                for value in message.values():
                    self.tokenizer.apply_chat_template(
                        [{"role": value, "content": "test"}], 
                        tokenize=False
                    )
            return True
        except:
            return False