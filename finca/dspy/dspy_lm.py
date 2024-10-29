import torch
import dspy

class DSPyLM(dspy.LM):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs # required dspy attribute

    def __call__(self, prompt=None, messages=None, **kwargs):
        # if i use the adapter then i get the prompt back
        # if i don't use the adapter then i get messages back
        
        # i think i want to not use the adapter, keep things simple, get them running, 
        # then try to optimize and see if I can get a better score on mmlu 
        # so i've got to get this working. then add examples (once i've pushed 0-shot to the max)
                                                            
        # I also need to make sure things work when not using dspy, now that the architecture is in an alright place


        # Handle messages if provided
        if messages:
            chat_template_supported = self._check_chat_template(messages)
            if chat_template_supported:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = prompt or "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)

        # Tokenize and ensure tensors are on the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
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