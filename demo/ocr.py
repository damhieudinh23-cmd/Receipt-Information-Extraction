
from pathlib import Path

import cv2
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "detected"
OUTPUT_DIR = BASE_DIR / "ocr_output"


def load_model():
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cpu"
    config["predictor"]["beamsearch"] = False

    return Predictor(config)


def recognize_text(image_path, predictor):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return ""

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    try:
        text = predictor.predict(pil_image)
        return text.strip()
    except Exception as e:
        print(f"Lỗi OCR {image_path.name}: {e}")
        return ""


def process_invoice(invoice_dir, predictor):
    invoice_dir = Path(invoice_dir)
    output_file = OUTPUT_DIR / f"{invoice_dir.name}.txt"

    crop_paths = sorted(invoice_dir.glob("crop_*.png"))

    if not crop_paths:
        print(f"Không tìm thấy ảnh crop trong {invoice_dir}")
        return

    results = []

    for crop_path in crop_paths:
        text = recognize_text(crop_path, predictor)

        print(f"{crop_path.name}: {text}")
        results.append(f"{crop_path.name}: {text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(results), encoding="utf-8")

    print(f"\nĐã lưu kết quả: {output_file}")


def main():
    if not INPUT_DIR.exists():
        print(f"Không tìm thấy thư mục: {INPUT_DIR}")
        return

    invoice_dirs = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_dir()
    )

    if not invoice_dirs:
        print(f"Không tìm thấy thư mục hóa đơn trong {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang tải mô hình VietOCR...")
    predictor = load_model()
    print("Đã tải mô hình thành công.\n")

    for invoice_dir in invoice_dirs:
        print(f"\n{'=' * 50}")
        print(f"Hóa đơn: {invoice_dir.name}")
        print("=" * 50)

        process_invoice(invoice_dir, predictor)


if __name__ == "__main__":
    main()

