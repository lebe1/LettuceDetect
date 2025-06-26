import re
import torch
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


# Making use of a custom method to be able to count appearences instead of simple regex lines
def is_valid_text(text):
    # Sample Rule: no word should appear more than 2 times (case-insensitive)
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    return all(count <= 2 for count in counts.values())


# StoppingCriteria that uses a validator function instead of a regex string
class StopOnInvalidText(StoppingCriteria):
    def __init__(self, validator_fn, tokenizer):
        self.validator_fn = validator_fn
        self.tokenizer = tokenizer
        self.stream = ""

    def reset(self):
        self.stream = ""

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token
        generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
        self.stream += generated

        print(generated, end="", flush=True)

        # Use custom validator function
        is_valid = self.validator_fn(self.stream)
        if not is_valid:
            print("\n[!] Invalid text detected by validator. Stopping generation.")
            return True
        return False


def test_stopping_with_validator():
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use advanced validator here
    stopping_criteria = StopOnInvalidText(is_valid_text, tokenizer)

    prompt = "The quick brown fox jumps"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("\n--- Generating ---\n")

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    print("\n--- Final Output ---")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_stopping_with_validator()
