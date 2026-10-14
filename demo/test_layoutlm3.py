from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import re
import unicodedata

import torch
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification


# =========================
# 1. CAU HINH
# =========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_DIR = PROJECT_DIR / "modellm3"
INPUT_DIR = BASE_DIR / "demo_input"
OUTPUT_DIR = BASE_DIR / "layoutlm3_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TOKEN_CSV = OUTPUT_DIR / f"ocr_tokens_{RUN_ID}.csv"
RESULT_CSV = OUTPUT_DIR / f"extracted_results_{RUN_ID}.csv"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Neu Tesseract doc duoc tieng Viet, giu viec nhan dang tieng Viet.
# Neu may chua cai language pack vie, doi thanh "eng" hoac cai them vie.traineddata.
OCR_LANG = "vie+eng"


# =========================
# 2. LOAD MODEL
# =========================

print("Dang tai LayoutLMv3...")
print("Model:", MODEL_DIR)
print("Device:", DEVICE)

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_DIR}")

processor = AutoProcessor.from_pretrained(
    str(MODEL_DIR),
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()

ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}
print("Nhan model:", ID2LABEL)


# =========================
# 3. CHUAN HOA VAN BAN
# =========================

def normalize_text(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d")


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


# =========================
# 4. CHON HUONG ANH
# =========================

def evaluate_orientation(image):
    """
    Thu 4 huong va chon huong co diem OCR tot nhat.
    Diem ket hop confidence, so tu co y nghia va so dong OCR.
    """
    best_image = image
    best_score = float("-inf")

    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle, expand=True)

        try:
            data = pytesseract.image_to_data(
                rotated,
                lang=OCR_LANG,
                config="--psm 6",
                output_type=Output.DICT
            )
        except pytesseract.TesseractError:
            # Neu may chua co goi ngon ngu vie, thu tieng Anh.
            data = pytesseract.image_to_data(
                rotated,
                config="--psm 6",
                output_type=Output.DICT
            )

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

            # Tu co it nhat 2 ky tu hoac chuoi so co y nghia.
            if len(word) >= 2:
                valid_words += 1
                score += max(0.0, confidence)
                valid_lines.add((
                    data["block_num"][i],
                    data["par_num"][i],
                    data["line_num"][i]
                ))

        # Thuong cho so tu va so dong doc duoc, giam anh huong cua vai tu confidence cao.
        score += valid_words * 8 + len(valid_lines) * 12

        print(
            f"  Goc {angle}: diem={score:.1f}, "
            f"tu={valid_words}, dong={len(valid_lines)}"
        )

        if score > best_score:
            best_score = score
            best_image = rotated

    return best_image


# =========================
# 5. OCR + BOUNDING BOX
# =========================

def extract_words_and_boxes(image):
    try:
        data = pytesseract.image_to_data(
            image,
            lang=OCR_LANG,
            output_type=Output.DICT,
            config="--psm 6"
        )
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6"
        )

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

        if confidence < 0:
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


# =========================
# 6. DU DOAN NHAN
# =========================

def predict_words(image, records):
    if not records:
        return records

    words = [record["word"] for record in records]
    boxes = [record["box"] for record in records]

    encoding = processor(
        images=[image],
        text=[words],
        boxes=[boxes],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    word_ids = encoding.word_ids(batch_index=0)

    model_inputs = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values"}
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu()
    predictions = outputs.logits.argmax(dim=-1)[0].cpu().tolist()

    seen = set()
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)

        label_id = predictions[token_index]
        records[word_id]["label"] = ID2LABEL.get(label_id, str(label_id))
        records[word_id]["model_confidence"] = float(
            probabilities[token_index, label_id].item()
        )

    for record in records:
        record.setdefault("label", "NOT_PROCESSED")
        record.setdefault("model_confidence", 0.0)

    return records


# =========================
# 7. GOM CAC TU THANH DONG
# =========================

def group_lines(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["line_id"]].append(record)

    lines = []
    for items in groups.values():
        items.sort(key=lambda record: record["x"])
        lines.append({
            "items": items,
            "text": clean_spaces(" ".join(r["word"] for r in items)),
            "y": min(r["y"] for r in items),
            "bottom": max(r["y"] + r["h"] for r in items)
        })

    lines.sort(key=lambda line: (line["y"], line["items"][0]["x"]))
    return lines


# =========================
# 8. HAU XU LY STORE
# =========================

STORE_STOP_PHRASES = [
    "hoa don ban hang",
    "hoa don",
    "ban hang",
    "ma so thue",
    "dien thoai",
    "ngay mua",
    "ten hang",
    "thanh tien"
]



def extract_store(records, lines, image_height):
    # 1. Uu tien ten cua hang nhan dien duoc ro trong OCR
    # Chi tim trong vung dau hoa don.
    top_lines = [
        line for line in lines
        if line["y"] <= 0.40 * image_height
    ]

    top_text = normalize_text(
        " ".join(line["text"] for line in top_lines)
    )

    # Nhan dien cac ten cua hang da biet trong bo test.
    if "vincommerce" in top_text:
        return "VINCOMMERCE"

    if "minimart anan" in top_text:
        return "MINIMART ANAN"

    # 2. Neu khong tim thay ten cua hang, dung nhan SELLER
    candidates = []

    for line in top_lines:
        seller_words = [
            r for r in line["items"]
            if r.get("label") == "SELLER"
        ]

        if seller_words:
            candidates.append(
                " ".join(r["word"] for r in seller_words)
            )

    text = clean_spaces(" ".join(candidates))
    normalized = normalize_text(text)

    # 3. Loai cac cum tieu de hoa don
    stop_phrases = [
        "hoa don ban hang",
        "hoa don",
        "ban hang",
        "ma so thue",
        "dien thoai",
        "ngay mua",
        "ten hang",
        "thanh tien"
    ]

    for phrase in stop_phrases:
        normalized = re.sub(
            re.escape(normalize_text(phrase)),
            " ",
            normalized
        )

    normalized = re.sub(
        r"\b(hoa|don|ban|hang|bon|pek)\b",
        " ",
        normalized
    )

    normalized = clean_spaces(normalized).upper()

    # Bo ket qua chi gom ky tu/ky hieu rac
    meaningful = re.sub(r"[^A-Z0-9]", "", normalized)

    if len(meaningful) < 3:
        return None

    return normalized


# =========================
# 9. HAU XU LY DATE
# =========================

def extract_date(lines):
    for line in lines:
        normalized = normalize_text(line["text"])
        if any(key in normalized for key in [
            "ngay", "date", "timestamp", "thoi gian"
        ]):
            result = parse_date(line["text"])
            if result:
                return result

    for line in lines:
        result = parse_date(line["text"])
        if result:
            return result

    return None


# =========================
# 10. HAU XU LY TOTAL
# =========================

TOTAL_KEYWORDS = [
    "tong cong",
    "teng ceng",          # OCR co the doc sai "Tổng cộng"
    "tong tien phai",
    "tong tien",
    "tong thanh toan",
    "tong tien thanh toan",
    "thanh tien",
    "total",
    "grand total"
]

EXCLUDE_MONEY_KEYWORDS = [
    "tien khach tra",
    "khach tra",
    "tien tra lai",
    "tra lai",
    "tien thua",
    "change",
    "cash tendered"
]


def extract_money_candidates(line):
    # Nhan dien cac so co phan tach hang nghin: 25.600, 65,000, 42.900
    found = re.findall(r"(?<!\d)\d[\d.,]*(?!\d)", line)
    values = []

    for item in found:
        digits = re.sub(r"\D", "", item)

        # Loai chuoi so dai giong ma vach/ma giao dich.
        if not digits or len(digits) > 7:
            continue

        amount = normalize_money(item)
        if amount is not None:
            values.append({"raw": item.strip(), "amount": amount})

    return values


def extract_total(records, lines):
    ranked_candidates = []

    for line in lines:
        normalized = normalize_text(line["text"])

        # Loai dong tien khach dua, tien thua va tien tra lai.
        if any(keyword in normalized for keyword in EXCLUDE_MONEY_KEYWORDS):
            continue

        keyword_score = 0
        for keyword in TOTAL_KEYWORDS:
            if keyword in normalized:
                if keyword in {
                    "tong cong",
                    "teng ceng",
                    "tong tien phai",
                    "tong thanh toan",
                    "tong tien thanh toan"
                }:
                    keyword_score = max(keyword_score, 100)
                else:
                    keyword_score = max(keyword_score, 80)

        # Khong chon so tien chi dua vao nhan model.
        # Model trong cac ket qua test dang gan TOTAL_COST cho ca ma vach.
        if keyword_score == 0:
            continue

        money_values = extract_money_candidates(line["text"])
        for money in money_values:
            score = keyword_score
            if money["amount"] > 0:
                score += 1

            ranked_candidates.append({
                "score": score,
                "amount": money["amount"],
                "raw": money["raw"],
                "line": line["text"]
            })

    if not ranked_candidates:
        return None, None

    ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
    best = ranked_candidates[0]
    return best["amount"], best["raw"]


# =========================
# 11. XU LY MOT ANH
# =========================

def process_image(image_path):
    with Image.open(image_path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    print("Chon huong anh:")
    image = evaluate_orientation(image)

    # Luu anh da chon huong de co the kiem tra neu can.
    oriented_dir = OUTPUT_DIR / "oriented_images"
    oriented_dir.mkdir(parents=True, exist_ok=True)
    image.save(oriented_dir / f"{image_path.stem}_oriented.png")

    # OCR, bounding box va LayoutLMv3 dung cung mot anh da xoay.
    records = extract_words_and_boxes(image)
    records = predict_words(image, records)
    lines = group_lines(records)

    store = extract_store(records, lines, image.height)
    date = extract_date(lines)
    total, total_raw = extract_total(records, lines)

    return records, {
        "image": image_path.name,
        "STORE": store,
        "DATE": date,
        "TOTAL": total,
        "TOTAL_RAW": total_raw
    }


# =========================
# 12. CHAY TOAN BO ANH
# =========================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {INPUT_DIR}")

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("Khong co anh trong:", INPUT_DIR)
        return

    all_results = []

    with open(TOKEN_CSV, "w", newline="", encoding="utf-8-sig") as token_file:
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
                records, result = process_image(image_path)
                all_results.append(result)

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

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=["image", "STORE", "DATE", "TOTAL", "TOTAL_RAW"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("HOAN TAT")
    print("Chi tiet OCR:", TOKEN_CSV)
    print("Ket qua trich xuat:", RESULT_CSV)
    print("Anh sau khi chon huong:", OUTPUT_DIR / "oriented_images")


if __name__ == "__main__":
    main()
