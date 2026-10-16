import re
import pytesseract
from pytesseract import Output
from config import OCR_LANG
from utils import preprocess_for_ocr, is_valid_ocr_token


def _run_tesseract_data(ocr_img):
    try:
        return pytesseract.image_to_data(
            ocr_img,
            lang=OCR_LANG,
            config="--psm 6",
            output_type=Output.DICT
        )
    except pytesseract.TesseractError:
        return pytesseract.image_to_data(
            ocr_img,
            config="--psm 6",
            output_type=Output.DICT
        )


def evaluate_orientation(image, verbose=True):
    """Thử 4 góc xoay và chọn góc có điểm OCR cao nhất."""
    best_image = image
    best_score = float("-inf")
    best_angle = 0

    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        ocr_img = preprocess_for_ocr(rotated)
        data = _run_tesseract_data(ocr_img)

        score = 0.0
        valid_words = 0
        valid_lines = set()

        for i, word in enumerate(data["text"]):
            word = word.strip()
            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                continue

            if not word or confidence < 0:
                continue

            alnum = re.sub(r"[^a-zA-Z0-9À-ỹ]", "", word)
            if len(alnum) >= 2 and confidence >= 30:
                valid_words += 1
                score += max(0.0, confidence)
                valid_lines.add((
                    data["block_num"][i],
                    data["par_num"][i],
                    data["line_num"][i]
                ))

        score += valid_words * 8 + len(valid_lines) * 12

        if verbose:
            print(
                f"  Goc {angle}: diem={score:.1f}, "
                f"tu={valid_words}, dong={len(valid_lines)}"
            )

        if score > best_score:
            best_score = score
            best_image = rotated
            best_angle = angle

    return best_image, best_angle


def extract_words_and_boxes(image):
    """Trích xuất danh sách từ và chuẩn hóa tọa độ về thang 0-1000."""
    ocr_img = preprocess_for_ocr(image)
    data = _run_tesseract_data(ocr_img)

    width, height = image.size
    records = []

    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if not is_valid_ocr_token(text, confidence):
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        box = [
            max(0, min(1000, int(1000 * x / width))),
            max(0, min(1000, int(1000 * y / height))),
            max(0, min(1000, int(1000 * (x + w) / width))),
            max(0, min(1000, int(1000 * (y + h) / height))),
        ]

        records.append({
            "word": text,
            "box": box,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence_ocr": confidence,
            "line_id": (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i]
            )
        })

    return records