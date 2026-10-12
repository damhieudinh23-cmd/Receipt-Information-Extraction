
import os
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output
from rembg import remove


# Đường dẫn Tesseract trên Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Đường dẫn dựa trên vị trí file pre_process.py
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "demo_input"
OUTPUT_DIR = BASE_DIR / "demo_output"

# Đổi thành True nếu muốn xóa nền
REMOVE_BACKGROUND = False


def evaluate_orientation(image):
    best_image = image
    best_score = -1

    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle, expand=True)

        data = pytesseract.image_to_data(
            rotated,
            config="--psm 6",
            output_type=Output.DICT
        )

        score = 0

        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()

            try:
                confidence = float(conf)
            except (ValueError, TypeError):
                continue

            if confidence > 50 and len(word) > 1:
                score += confidence

        print(f"  Góc {angle}: điểm = {score}")

        if score > best_score:
            best_score = score
            best_image = rotated

    return best_image


def preprocess_image(input_path, output_path,
                     remove_background=False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Đọc ảnh và sửa hướng theo EXIF
    with Image.open(input_path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    print(f"\nĐang xử lý: {input_path.name}")

    # 2. Tìm hướng ảnh phù hợp nhất
    image = evaluate_orientation(image)

    # 3. Xóa nền nếu được yêu cầu
    if remove_background:
        image = remove(image).convert("RGB")

    # 4. Tạo thư mục đầu ra và lưu ảnh
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Đã lưu: {output_path}")
    return str(output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ) if INPUT_DIR.exists() else []

    if not image_paths:
        print(f"Không tìm thấy ảnh trong: {INPUT_DIR}")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh.")

    for input_path in image_paths:
        output_path = OUTPUT_DIR / f"{input_path.stem}_processed.png"

        try:
            preprocess_image(
                input_path,
                output_path,
                remove_background=REMOVE_BACKGROUND
            )
        except Exception as e:
            print(f"Lỗi khi xử lý {input_path.name}: {e}")

    print("\nHoàn tất tiền xử lý!")


if __name__ == "__main__":
    main()