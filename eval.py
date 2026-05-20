# eval.py
from sample import *   # brings in model, ctx, encode, decode, etc.
import json

def eval():
    
    with open("eval_data.json", "r") as f:
        data = json.load(f)
    
    with torch.no_grad():
        with ctx:
            for pair in data:
                x = torch.tensor(encode(pair["prompt"]), dtype=torch.long, device=device)[None, ...]
                fixed_response = encode(pair["response"])
                y, seq_prob = model.generate(x,
                                             max_new_tokens,
                                             temperature=temperature,
                                             top_k=top_k,
                                             show_probs=show_probs,
                                             decode=decode,
                                             fixed_response=fixed_response)
                print(f"Prompt: {pair['prompt']}")
                print(f"Response: {decode(y[0].tolist())}")
                print(f"\nProbability of response: {seq_prob}")
                print('-' * 32)

eval()