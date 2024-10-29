import torch
import dspy

class DSPyLM(dspy.LM):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs # required dspy attribute

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Handle messages if provided
        if messages:
            chat_template_supported = self._check_chat_template(messages)
            if chat_template_supported:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = prompt or "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
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
        """Check if chat template is supported for these messages"""
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