import re
import json

from lettucedetect.detectors.number import NumberDetector

def simulate_llm_generation(batch, detector: NumberDetector):
    context = [batch["prompt"]]
    gold_answer = batch["answer"]
    tokens = re.findall(r'\b\w+\b|\S', gold_answer)  # Better token splitting

    # print(f"\nSimulating generation for:\nPrompt: {batch['prompt']}\n")
    # generated = ""

    for idx, token in enumerate(tokens):
        token_preds = detector.predict(context, token, output_format="tokens")
        hallucinated = [pred for pred in token_preds if pred['pred'] == 1]
        if hallucinated:
            print(f"[Step {idx+1}] Token: '{token}' -> Hallucinated Number Detected!")
        else:
            print(f"[Step {idx+1}] Token: '{token}'")

    # print("\nFinal Answer Generated:\n", generated.strip())

if __name__ == "__main__":
    # Load data
    with open("../data/ragtruth/hallucinated_numbers_only.json", "r") as f:
        dataset = json.load(f)

    # Instantiate the hallucination detector
    detector = NumberDetector()

    # Simulate LLM generation on one example for debugging
    simulate_llm_generation(dataset[2], detector)


    # Run over every experiment 
    # Suggestion to comment else-statement in method and add break in if-statement
    # for batch in dataset: 
    #     simulate_llm_generation(batch, detector)