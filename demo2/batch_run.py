from datetime import datetime
import csv
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_EXTENSIONS
from model_engine import LayoutLMv3Predictor
from pipeline import process_receipt


def main():
    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("Khong co anh trong:", INPUT_DIR)
        return

    predictor = LayoutLMv3Predictor()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    token_csv = OUTPUT_DIR / f"ocr_tokens_{run_id}.csv"
    result_csv = OUTPUT_DIR / f"extracted_results_{run_id}.csv"

    all_results = []

    with open(token_csv, "w", newline="", encoding="utf-8-sig") as token_file:
        token_writer = csv.DictWriter(
            token_file,
            fieldnames=[
                "image", "word", "predicted_label",
                "model_confidence", "ocr_confidence",
                "x", "y", "w", "h",
                "x0_norm", "y0_norm", "x1_norm", "y1_norm"
            ]
        )
        token_writer.writeheader()

        for index, image_path in enumerate(image_paths, 1):
            print("\n" + "=" * 70)
            print(f"[{index}/{len(image_paths)}] {image_path.name}")
            print("=" * 70)

            try:
                output = process_receipt(image_path, predictor)
                result = output["result"]
                records = output["records"]

                all_results.append({
                    "image": result["image"],
                    "STORE": result["STORE"],
                    "DATE": result["DATE"],
                    "TOTAL": result["TOTAL"],
                    "TOTAL_RAW": result["TOTAL_RAW"]
                })

                for record in records:
                    token_writer.writerow({
                        "image": image_path.name,
                        "word": record["word"],
                        "predicted_label": record["label"],
                        "model_confidence": round(record["model_confidence"], 4),
                        "ocr_confidence": round(record["confidence_ocr"], 2),
                        "x": record["x"],
                        "y": record["y"],
                        "w": record["w"],
                        "h": record["h"],
                        "x0_norm": record["box"][0],
                        "y0_norm": record["box"][1],
                        "x1_norm": record["box"][2],
                        "y1_norm": record["box"][3]
                    })

                print("STORE:", result["STORE"])
                print("DATE:", result["DATE"])
                print("TOTAL:", result["TOTAL"])
                print("TOTAL_RAW:", result["TOTAL_RAW"])

            except Exception as error:
                print("Loi:", error)
                all_results.append({
                    "image": image_path.name,
                    "STORE": None,
                    "DATE": None,
                    "TOTAL": None,
                    "TOTAL_RAW": None
                })

    with open(result_csv, "w", newline="", encoding="utf-8-sig") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=["image", "STORE", "DATE", "TOTAL", "TOTAL_RAW"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("HOAN TAT")
    print("Chi tiet OCR:", token_csv)
    print("Ket qua trich xuat:", result_csv)


if __name__ == "__main__":
    main()