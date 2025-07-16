import json
import re
from word2number import w2n

def extract_cardinal_digits(text):
    return re.findall(r'\b\d+\b', text)

def extract_number_words(text):
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
    digit_numbers = extract_cardinal_digits(prompt)
    word_numbers = extract_number_words(prompt)
    return set(digit_numbers + word_numbers)

def process_summary_item(item):
    if item.get("task_type") != "Summary":
        return None

    prompt_numbers = normalize_prompt_numbers(item.get("prompt", ""))
    answer_numbers = extract_cardinal_digits(item.get("answer", ""))

    hallucinated = [num for num in answer_numbers if num not in prompt_numbers]

    if hallucinated:
        item["hallucinated_numbers"] = hallucinated
        return item
    else:
        return None

def main(input_path, json_output_path, stats_output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_items = 0
    data2txt_items = 0
    qa_items = 0
    hallucinated_items = []

    for item in data:
        task_type = item.get("task_type")
        if task_type == "Summary":
            summary_items += 1
            result = process_summary_item(item)
            if result:
                hallucinated_items.append(result)
        elif task_type == "Data2txt":
            data2txt_items += 1
        elif task_type == "QA":
            qa_items += 1
        else:
            print("Incorrect task type: ", task_type)

    total_items = summary_items + data2txt_items + qa_items

    # Save hallucinated items to JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(hallucinated_items, f, indent=4, ensure_ascii=False)

    # Save stats to TXT file
    with open(stats_output_path, "w", encoding="utf-8") as f:
        f.write(f"Summary entries: {summary_items}\n")
        f.write(f"Data2txt entries: {data2txt_items}\n")
        f.write(f"QA entries : {qa_items}\n")
        f.write(f"Total entries : {total_items}\n")
        f.write(f"Total entries with hallucinated numbers: {len(hallucinated_items)}\n")

    # Also print to console
    print(f"Summary entries: {summary_items}")
    print(f"Data2txt entries: {data2txt_items}")
    print(f"QA entries : {qa_items}")
    print(f"Total entries : {total_items}")
    print(f"Total entries with hallucinated numbers: {len(hallucinated_items)}")

if __name__ == "__main__":
    input_file = "../data/ragtruth/ragtruth_data.json"
    json_output_file = "../data/ragtruth/output_with_hallucinations.json"
    stats_output_file = "../data/ragtruth/hallucination_stats.txt"
    main(input_file, json_output_file, stats_output_file)

