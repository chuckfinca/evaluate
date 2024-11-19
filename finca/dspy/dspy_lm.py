import torch
import dspy

class DSPyLM(dspy.LM):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        
        generation_kwargs = {
                'max_new_tokens': 100,
                'pad_token_id': self.tokenizer.eos_token_id,
                'do_sample': False,  # This is all you need for pure greedy decoding (i.e. it will deterministically pick the most likely token)
                'temperature': None, # required for do_sample=False
                'top_p': None, # required for do_sample=False
                **kwargs
            }
        
        self.kwargs = generation_kwargs # required dspy attribute

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Handle messages if provided
        if messages:
            chat_template_supported = self._check_chat_template(messages)
            if chat_template_supported:
                print("using chat template")
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                print("NOT using chat template")
                formatted_prompt = prompt or "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generation_kwargs = {
                'max_new_tokens': 100,
                'pad_token_id': self.tokenizer.eos_token_id,
                'do_sample': False,  # This is all you need for pure greedy decoding (i.e. it will deterministically pick the most likely token)
                'temperature': None, # required for do_sample=False
                'top_p': None, # required for do_sample=False
                **kwargs
            }
            output = self.model.generate(**inputs, **generation_kwargs)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=False)
            
            # this is a bit of a hack that seems to work.
            # we remove the prompt using its character count
            # however it doesn't remove everything we want because
            # a piece is added during generation to set up the 
            # role / turn of the assistant.
            # Regardless, DSPy seems to be able to extract from the result
            result = decoded_output[len(formatted_prompt):]
            return [result]

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