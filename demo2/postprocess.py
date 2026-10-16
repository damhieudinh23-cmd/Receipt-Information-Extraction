from collections import defaultdict
import re
from config import (
    STORE_STOP_PHRASES,
    TOTAL_KEYWORDS,
    EXCLUDE_MONEY_KEYWORDS
)
from utils import (
    normalize_text,
    clean_spaces,
    parse_date,
    normalize_money
)


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


def clean_store_candidate(raw_text):
    normalized = normalize_text(raw_text)

    for phrase in STORE_STOP_PHRASES:
        normalized = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", normalized)

    normalized = re.sub(
        r"\b(hoa|don|ban|hang|bon|pek|ngay|quay)\b",
        " ",
        normalized
    )

    cleaned = re.sub(r"[^a-z0-9+\s]", " ", normalized).upper()
    cleaned = clean_spaces(cleaned)

    tokens = [
        tok for tok in cleaned.split()
        if len(tok) >= 2 or tok in {"+"}
    ]
    cleaned = " ".join(tokens)

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

    # 1. Nhận diện chuỗi cửa hàng quen thuộc
    if (
        re.search(r"vi\s*n?\s*c\s*o\s*m\s*m?\s*e\s*r\s*c\s*e", top_text)
        or re.search(r"\bvm\s*\+", top_text)
        or "vinmart.com" in full_text
        or "winmart" in full_text
    ):
        return "VINCOMMERCE"

    if re.search(r"mini\s*mart\s*an\s*an", top_text):
        return "MINIMART ANAN"

    # 2. Lấy từ nhãn SELLER của model
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

    # 3. Fallback: Dòng hợp lệ đầu tiên ở đầu hóa đơn
    for line in top_lines[:4]:
        line_norm = normalize_text(line["text"])
        if any(stop in line_norm for stop in ["hoa don", "ngay ban", "ngay mua"]):
            break
        candidate = clean_store_candidate(line["text"])
        if candidate:
            return candidate

    return None


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
            score = keyword_score + (1 if money["amount"] > 0 else 0)
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