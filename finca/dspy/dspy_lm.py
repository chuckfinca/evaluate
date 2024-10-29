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
        inputs = None
        if messages:
            print("messages is:")
            print(f" - a {type(messages)}")
            print(f" - len = {len(messages)}")
            chat_template_supported = True
            for index, message in enumerate(messages):
                print(f"-message {index} is:")
                print(f"  - a {type(message)}")
                print(f"  - len = {len(message)}")
                for key, value in messages[0].items():
                    print(f"   - key={key}: value=|||{value}|||")
                    try:
                        self.tokenizer.apply_chat_template([{"role": value, "content": "test"}], tokenize=False)
                        print(f"Chat template role {value} supported!")
                    except:
                        chat_template_supported = False
                        
        if chat_template_supported:
            print("!!!!!!!!Applying chat template!!!!!!!!")
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            if not prompt:
                prompt = "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generation_config = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 100,
            "do_sample": False,  # This is all you need for pure greedy decoding
            "temperature": None, # required for do_sample=False
            "top_p": None # required for do_sample=False
            }   
            output = self.model.generate(**inputs, **generation_config)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print("----------------------------------------------------------------------------------------------------------------")
            print("prompt:")
            print(prompt)
            print("----------------------------------------------------------------------------------------------------------------")
            print("decoded_output:")
            print(decoded_output)
            return decoded_output
