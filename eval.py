from sample import *   # brings in model, ctx, encode, decode, etc.
import json
from datasets import load_dataset

dataset = load_dataset("Rowan/hellaswag", split="validation")

# Create prompt-response pairs from the dataset
pairs = []
for row in dataset.select(range(100)):
    endings = row["endings"]
    correct = int(row["label"])
    
    prompt = (
        f"{row['ctx']}\n"  # Implicit string concatenation without "+"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n"
        f"Which ending makes the most sense?\nAnswer:"
    )
    response = [" A", " B", " C", " D"][correct]  # Direct list indexing: response is A-D based on whether correct is 0-3
    pairs.append({"prompt": prompt, "response": response})  # Add one dictionary to list

# Write the prompt-response pairs to the json file
with open("eval_data.json", "w") as f:
    json.dump(pairs, f, indent=2)
    
    
def eval():
    
    seq_probs = []
    
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
                seq_probs.append(seq_prob)
                print(f"Prompt:\n{pair['prompt']}")
                print(f"Response:\n{pair["response"]}")
                print(f"Probability of response: {seq_prob}")
                print('-' * 32)
                
    print(f"\nAvg response probability: {sum(seq_probs)/len(seq_probs)}")

eval()