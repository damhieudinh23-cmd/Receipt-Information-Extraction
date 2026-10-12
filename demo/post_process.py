import json
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent / "extract_output"
OUTPUT_DIR = Path(__file__).resolve().parent / "post_process_output"

KEEP_FIELDS = ["source", "STORE", "DATE", "TOTAL"]

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        key: data[key]
        for key in KEEP_FIELDS
        if key in data
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for input_path in INPUT_DIR.glob("*.json"):
        output_path = OUTPUT_DIR / input_path.name
        process_file(input_path, output_path)
        print(f"Đã xử lý: {output_path.name}")

if __name__ == "__main__":
    main()