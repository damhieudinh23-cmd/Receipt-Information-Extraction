from datetime import datetime
import re
import unicodedata
from PIL import ImageOps, ImageFilter


def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )


def clean_spaces(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_date(text):
    match = re.search(
        r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)",
        text
    )
    if not match:
        return None

    day, month, year = map(int, match.groups())
    try:
        return datetime(year, month, day).strftime("%d/%m/%Y")
    except ValueError:
        return None


def normalize_money(text):
    digits = re.sub(r"\D", "", str(text))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def preprocess_for_ocr(image):
    """Tăng tương phản và làm nét chữ in nhiệt bị mờ cho Tesseract."""
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    return contrasted.filter(ImageFilter.SHARPEN)


def is_valid_ocr_token(word, confidence):
    """Lọc bỏ các ký tự rác sinh ra từ nền ảnh."""
    cleaned = clean_spaces(word)
    if not cleaned:
        return False

    alnum = re.sub(r"[^a-zA-Z0-9À-ỹ]", "", cleaned)
    if not alnum:
        return False

    if len(alnum) == 1 and confidence < 45:
        return False

    if confidence < 20:
        return False

    return True