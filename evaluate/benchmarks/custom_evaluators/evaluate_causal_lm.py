import torch
import numpy as np

choices = ["A", "B", "C", "D"]

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    for j, choice in enumerate(choices):
        prompt += f"\n{choice}. {df.iloc[idx, j+1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, 5]}"
    return prompt

def format_prompt(train_df, test_df, test_idx):
    prompt = "Answer the following multiple choice questions. Choose the best answer from A, B, C, or D.\n\n"
    for i in range(len(train_df)):
        prompt += format_example(train_df, i) + "\n\n"
    prompt += format_example(test_df, test_idx, include_answer=False)
    return prompt

def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    preds = []
    probs = []

    for i in range(len(test_df)):
        prompt = format_prompt(dev_df, test_df, i)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1]
        probs_i = torch.nn.functional.softmax(logits, dim=-1)
        
        choice_probs = [probs_i[tokenizer.encode(choice, add_special_tokens=False)[0]].item() for choice in choices]
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_probs)]
        
        probs.append(choice_probs)
        preds.append(pred)
        cors.append(pred == test_df.iloc[i, 5])

    acc = np.mean(cors)
    print(f"{subject} Accuracy: {acc:.3f}")

    return cors, acc, probs