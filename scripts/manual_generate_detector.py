import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "microsoft/phi-2"
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

def extract_numbers(text):
    import re
    pattern = r'\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b'
    return set(re.findall(pattern, text))

def manual_generate(prompt, context, max_new_tokens=100):
    context_numbers = extract_numbers(context.lower() + prompt.lower())
    input_text = f"Answer the following question based on the given context.\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)["input_ids"]

    generated = input_ids
    past_key_values = None
    hallucinated_numbers = set()
    # full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # Greedy decode 
        next_token_id = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_token_id.unsqueeze(-1)], dim=-1)

        # Print new token info
        token_index = generated.shape[-1] - 1
        token_str = tokenizer.decode(generated[0, token_index])
        print(f"Token {token_index}: {generated[0, token_index].item()} -> '{token_str}'")


        # Sampling decode
        # probs = torch.softmax(logits, dim=-1)
        # next_token_id = torch.multinomial(probs, num_samples=1)


        # Decode current full text
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Check hallucinated numbers
        new_numbers = extract_numbers(full_text)
        hallucinations = new_numbers - context_numbers

        # Stop if a hallucinated number is generated
        if hallucinations:
            hallucinated_numbers.update(hallucinations)
            break

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    log = "\n\n[Hallucinated Numbers Detected]: " + ", ".join(sorted(hallucinated_numbers)) if hallucinated_numbers else "\n\n[No Hallucinated Numbers Detected]"
    return decoded + log

