import json

def main(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = 0
    for _ in data:
        items += 1

    print(f"Total entries : {items}")

if __name__ == "__main__":
    input_file = "../data/ragtruth/ragtruth_data.json"   

    main(input_file)