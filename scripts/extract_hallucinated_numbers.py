import json
import re
from word2number import w2n

def extract_cardinal_digits(text):
    """Extract digit-based cardinal numbers from text."""
    return re.findall(r'\b\d+\b', text)

def extract_number_words(text):
    """
    Extract contiguous sequences of words that could represent numbers.
    Example: "twenty one", "one hundred and five"
    """
    # Basic pattern for number words
    number_word_pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                                     r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                                     r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                                     r'eighty|ninety|hundred|thousand|million|billion|and|[-])+\b',
                                     re.IGNORECASE)

    matches = number_word_pattern.finditer(text)
    number_strings = []
    for match in matches:
        phrase = match.group().replace("-", " ").lower()
        try:
            number = str(w2n.word_to_num(phrase))
            number_strings.append(number)
        except ValueError:
            continue
    return number_strings

def normalize_prompt_numbers(prompt):
    """Return a set of all numeric digits and word-number equivalents in the prompt."""
    digit_numbers = extract_cardinal_digits(prompt)
    word_numbers = extract_number_words(prompt)
    return set(digit_numbers + word_numbers)

def process_summary_item(item):
    if item.get("task_type") != "Summary":
        return None  # Skip non-summary tasks

    prompt_numbers = normalize_prompt_numbers(item.get("prompt", ""))
    answer_numbers = extract_cardinal_digits(item.get("answer", ""))

    hallucinated = [num for num in answer_numbers if num not in prompt_numbers]

    if hallucinated:
        item["hallucinated_numbers"] = hallucinated
        return item
    else:
        return None  # Skip items without hallucinated numbers

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hallucinated_items = []
    for item in data:
        result = process_summary_item(item)
        if result:
            hallucinated_items.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hallucinated_items, f, indent=4, ensure_ascii=False)

    print(f"Total entries with hallucinated numbers: {len(hallucinated_items)}")

if __name__ == "__main__":
    input_file = "../data/ragtruth/ragtruth_data.json"   
    output_file = "../data/ragtruth/output_with_hallucinations2.json"
    main(input_file, output_file)
