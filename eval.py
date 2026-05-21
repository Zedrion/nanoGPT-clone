import importlib
import sample
importlib.reload(sample)  # forces sample to be fresh every time eval.py is called
from sample import *   # brings in model, ctx, encode, decode, etc.
import json
from datasets import load_dataset

# -----------------------------------------------------------------------------
eval_mode = "ranked"  # "prob" / "ranked"
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
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
    
    # Direct list indexing: response is A-D based on whether correct is 0-3
    # Leading space because there could be a some confusing stuff with the tokenization 
    # leading to incorrect token probability calculation
    # Work from a previous assignment task indicated that putting a trailing space in the prompt instead 
    # had some troubles as well
    response = [" A", " B", " C", " D"][correct]  
    
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

options = [" A", " B", " C", " D"]

def ranked_eval():
    
    results = []
    
    # Load the json file
    with open("eval_data.json", "r") as f:
        data = json.load(f)  # List of dictionaries
    
    with torch.no_grad():
        with ctx:
            for pair in data:
                option_probs = []  # List of 4 probabilities
                for option in options:
                    x = torch.tensor(encode(pair["prompt"]), dtype=torch.long, device=device)[None, ...]
                    fixed_response = encode(option)
                    print(f"fixed_response:{fixed_response}")
                    _, option_prob = model.generate(x,
                                                    max_new_tokens,
                                                    temperature=temperature,
                                                    top_k=top_k,
                                                    show_probs=show_probs,
                                                    decode=decode,
                                                    fixed_response=fixed_response)
                    option_probs.append(option_prob)
                
                # Answer is index of option with highest prob     
                answer = options[option_probs.index(max(option_probs))]
                print(f"Option probs: {option_probs}")
                results.append(answer==pair["response"])  # True if correct, False if not

                print(f"Prompt:\n{pair['prompt']}")
                print(f"Answer:\n{answer}")
                print(f"Correct answer:\n{pair["response"]}")
                print('-' * 32)
            
    acc = sum(results) / len(results)
    print(f"Accuracy: {acc*100:.2f}%")
    
def eval_select(eval_mode="prob"):
    
    match eval_mode:
        case "ranked":
            ranked_eval()
        case _:
            eval()

eval_select(eval_mode=eval_mode)