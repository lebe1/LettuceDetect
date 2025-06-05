from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
import torch
import re


class NumberHallucinationProcessor(LogitsProcessor):
    def __init__(self, context_text: str, tokenizer, penalty: float = -100.0):
        self.context_numbers = self._extract_numbers(context_text.lower())
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.generated_text = ""
        self.hallucinated_numbers = set()

    def _extract_numbers(self, text: str) -> set:
        number_pattern = r'\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b'
        return set(re.findall(number_pattern, text))

    def _detect_hallucinated_numbers(self, text: str) -> set:
        gen_nums = self._extract_numbers(text)
        return {n for n in gen_nums if n not in self.context_numbers}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.generated_text = decoded

        hallucinated = self._detect_hallucinated_numbers(decoded)
        self.hallucinated_numbers.update(hallucinated)

        for num in hallucinated:
            token_ids = self.tokenizer.encode(num, add_special_tokens=False)
            for tid in token_ids:
                if tid < scores.shape[-1]:
                    scores[0, tid] += self.penalty  # Penalize

        return scores


# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_with_live_penalty(prompt: str, context: str, max_new_tokens=100):
    input_text = f"Context: {context}\n\nQ: {prompt}\nA:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    processor = NumberHallucinationProcessor(
        context_text=context,
        tokenizer=tokenizer,
        penalty=-100.0
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=[processor],
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append hallucination log
    if processor.hallucinated_numbers:
        log = "\n\n[Hallucinated Numbers Detected]: " + ", ".join(sorted(processor.hallucinated_numbers))
    else:
        log = "\n\n[No Hallucinated Numbers Detected]"

    return decoded + log


# Example
context = "In 2000, 8 million citizens lived in Austria."
prompt = "How many live there in 2025?"

output = generate_with_live_penalty(prompt, context)
print(output)
