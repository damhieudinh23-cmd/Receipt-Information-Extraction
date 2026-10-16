from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
import csv
import re
import unicodedata

import torch
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from pytesseract import Output
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

# Ẩn cảnh báo FutureWarning từ thư viện transformers
warnings.filterwarnings("ignore", category=FutureWarning)


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
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text


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
    """
    Tang do tuong phan va lam ro net chu in nhiet bi mo cho Tesseract,
    nhung giu nguyen kich thuoc anh de khong lam lech bounding box.
    """
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    sharpened = contrasted.filter(ImageFilter.SHARPEN)
    return sharpened


def is_valid_ocr_token(word, confidence):
    """
    Loc bo cac ky tu rac sinh ra tu nen anh (nhu van chieu truc).
    """
    cleaned = clean_spaces(word)
    if not cleaned:
        return False

    # Ky tu don le khong phai chu/so (vi du: "{", "_", ".", "“")
    alnum = re.sub(r"[^a-zA-Z0-9À-ỹ]", "", cleaned)
    if not alnum:
        return False

    # Cac tu chi co 1 ky tu va confidence thap thuong la nhieu nen
    if len(alnum) == 1 and confidence < 45:
        return False

    # Cac tu confidence qua thap
    if confidence < 20:
        return False

    return True


# =========================
# 4. CHON HUONG ANH
# =========================

def evaluate_orientation(image):
    best_image = image
    best_score = float("-inf")

    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        ocr_img = preprocess_for_ocr(rotated)

        try:
            data = pytesseract.image_to_data(
                ocr_img,
                lang=OCR_LANG,
                config="--psm 6",
                output_type=Output.DICT
            )
        except pytesseract.TesseractError:
            data = pytesseract.image_to_data(
                ocr_img,
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
    ocr_img = preprocess_for_ocr(image)

    try:
        data = pytesseract.image_to_data(
            ocr_img,
            lang=OCR_LANG,
            output_type=Output.DICT,
            config="--psm 6"
        )
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(
            ocr_img,
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
    "hoa don thanh toan",
    "hoa don",
    "ban hang",
    "ma so thue",
    "dien thoai",
    "ngay mua",
    "ngay ban",
    "ten hang",
    "mat hang",
    "thanh tien",
    "quay"
]


def clean_store_candidate(raw_text):
    """
    Lam sach chuoi STORE va loai bo cac ky tu rac hoac tu don le vo nghia.
    """
    normalized = normalize_text(raw_text)

    for phrase in STORE_STOP_PHRASES:
        normalized = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", normalized)

    normalized = re.sub(
        r"\b(hoa|don|ban|hang|bon|pek|ngay|quay)\b",
        " ",
        normalized
    )

    # Chi giu lai chu cai, chu so, dau + va khoang trang
    cleaned = re.sub(r"[^a-z0-9+\s]", " ", normalized).upper()
    cleaned = clean_spaces(cleaned)

    # Loai bo cac chuoi chi gom cac ky tu don le roi rac nhu "A 4 A"
    tokens = [
        tok for tok in cleaned.split()
        if len(tok) >= 2 or tok in {"+"}
    ]
    cleaned = " ".join(tokens)

    # Bat buoc phai co it nhat 1 tu chua >= 3 chu cai lien tiep (hoac VM+)
    if not re.search(r"([A-Z]{3,}|VM\+)", cleaned):
        return None

    return cleaned


def extract_store(records, lines, image_height):
    top_lines = [
        line for line in lines
        if line["y"] <= 0.40 * image_height
    ]

    top_text = normalize_text(" ".join(line["text"] for line in top_lines))
    full_text = normalize_text(" ".join(line["text"] for line in lines))

    # 1. Nhan dien linh hoat cac chuoi cua hang pho bien (ke ca khi OCR doc khuyet net)
    if (
        re.search(r"vi\s*n?\s*c\s*o\s*m\s*m?\s*e\s*r\s*c\s*e", top_text)
        or re.search(r"\bvm\s*\+", top_text)
        or "vinmart.com" in full_text
        or "winmart" in full_text
    ):
        return "VINCOMMERCE"

    if re.search(r"mini\s*mart\s*an\s*an", top_text):
        return "MINIMART ANAN"

    # 2. Lay tu nhan SELLER cua LayoutLMv3 (chi lay token co do tin cay tot)
    seller_lines = []
    for line in top_lines:
        seller_words = [
            r["word"] for r in line["items"]
            if "SELLER" in str(r.get("label", "")).upper()
            and r.get("confidence_ocr", 0) >= 40
            and r.get("model_confidence", 0) >= 0.45
        ]
        if seller_words:
            candidate = clean_store_candidate(" ".join(seller_words))
            if candidate:
                seller_lines.append(candidate)

    if seller_lines:
        return clean_spaces(" ".join(seller_lines))

    # 3. Fallback: Tim dong hop le dau tien o phan dau hoa don (truoc tieu de HOA DON)
    for line in top_lines[:4]:
        line_norm = normalize_text(line["text"])
        if any(stop in line_norm for stop in ["hoa don", "ngay ban", "ngay mua"]):
            break
        candidate = clean_store_candidate(line["text"])
        if candidate:
            return candidate

    return None


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
    "tong tien phai t.toan",
    "tong tien phai",
    "tong cong",
    "teng ceng",
    "tong thanh toan",
    "tong tien thanh toan",
    "tong tien",
    "thanh tien",
    "total",
    "grand total"
]

EXCLUDE_MONEY_KEYWORDS = [
    "tien khach tra",
    "khach tra",
    "tien mat",
    "tien tra lai",
    "tra lai",
    "tien thua",
    "da giam",
    "a giam",
    "change",
    "cash tendered"
]


def extract_money_candidates(line):
    found = re.findall(r"(?<!\d)\d[\d.,]*(?!\d)", line)
    values = []

    for item in found:
        digits = re.sub(r"\D", "", item)

        if not digits or len(digits) > 8:
            continue

        amount = normalize_money(item)
        if amount is not None:
            values.append({"raw": item.strip(), "amount": amount})

    return values


def extract_total(records, lines):
    ranked_candidates = []

    for line in lines:
        normalized = normalize_text(line["text"])

        if any(keyword in normalized for keyword in EXCLUDE_MONEY_KEYWORDS):
            continue

        keyword_score = 0
        for keyword in TOTAL_KEYWORDS:
            if keyword in normalized:
                if keyword in {
                    "tong tien phai t.toan",
                    "tong tien phai",
                    "tong cong",
                    "teng ceng",
                    "tong thanh toan",
                    "tong tien thanh toan"
                }:
                    keyword_score = max(keyword_score, 100)
                else:
                    keyword_score = max(keyword_score, 80)

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

    oriented_dir = OUTPUT_DIR / "oriented_images"
    oriented_dir.mkdir(parents=True, exist_ok=True)
    image.save(oriented_dir / f"{image_path.stem}_oriented.png")

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