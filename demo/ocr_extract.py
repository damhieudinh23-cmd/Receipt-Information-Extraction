
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# 1. CAU HINH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "my_model"
INPUT_DIR = BASE_DIR / "ocr_output"
OUTPUT_DIR = BASE_DIR / "extract_output"

MAX_LENGTH = 256
OVERLAP = 48
DEBUG = True

TARGET_FIELDS = ("STORE", "DATE", "TOTAL", "PROD")

device = torch.device("cpu")


# ============================================================
# 2. TAI MO HINH
# ============================================================

def load_model():
    print("Dang tai mo hinh PhoBERT NER...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )
    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)
    model.eval()

    return tokenizer, model


# ============================================================
# 3. DOC VAN BAN OCR
# ============================================================

def read_ocr_file(file_path):
    """
    OCR output co dang:
    crop_0000.png: Vincommerce
    crop_0001.png: Ngay ban: 14/08/2020 19:55

    Chi xoa tien to ten file crop, giu nguyen noi dung OCR.
    """
    lines = []

    for line in file_path.read_text(
        encoding="utf-8"
    ).splitlines():

        text = re.sub(
            r"^\s*crop_\d+\.png:\s*",
            "",
            line,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

        if text:
            lines.append(text)

    return "\n".join(lines)


# ============================================================
# 4. CAC HAM HAU XU LY TU CODE GOC
# ============================================================

def get_ultimate_total(text):
    """
    Tim gia tien co dinh dang hang nghin trong 20 dong cuoi.
    Uu tien gia tri lon nhat trong phan cuoi hoa don.
    """
    footer = " ".join(text.splitlines()[-20:])

    matches = re.findall(
        r"(?<!\d)[1-9]\d{0,2}(?:[.,]\d{3})+(?!\d)",
        footer,
    )

    if not matches:
        return None

    return max(
        matches,
        key=lambda value: int(
            re.sub(r"[.,]", "", value)
        ),
    )


def get_ultimate_date(text):
    matches = re.findall(
        r"(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)",
        text,
    )

    if not matches:
        return None

    parts = re.split(r"[/.-]", matches[0])

    day = parts[0].zfill(2)
    month = parts[1].zfill(2)
    year = parts[2]

    return f"{day}/{month}/{year}"


def format_store_name(store_list):
    if not store_list:
        return "Khong tim thay"

    raw = "".join(store_list).replace(" ", "").lower()

    if "vin" in raw or "comerce" in raw or "commerce" in raw:
        return "Vincommerce"

    if "uni" in raw or "qlo" in raw:
        return "UNIQLO"

    if "mini" in raw or "anan" in raw:
        return "MINIMART ANAN"

    return raw.upper()


def clean_product_name(prod_str):
    text = prod_str.replace("@@", "").replace("<unk>", "")

    # Bo ma vach
    text = re.sub(r"\b\d{7,15}\b", " ", text)

    # Bo gia tien co dau cham/phay
    text = re.sub(r"\b\d{1,3}[.,]\d{3}\b", " ", text)

    # Bo gia tien bi tach bang khoang trang
    text = re.sub(r"\b\d{1,3}(?:\s+\d{3})+\b", " ", text)
    text = re.sub(r"\b\d(?:\s+\d){1,3}\s+\d{3}\b", " ", text)

    # Bo cac chuoi so dai
    text = re.sub(r"\b(?:\d+\s+){3,}\d+\b", " ", text)

    # Chi giu chu, so va mot so ky tu can thiet
    text = re.sub(
        r"[^a-zA-ZÀ-ỹ0-9\s%/-]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    # Xoa so le o cuoi ten san pham
    text = re.sub(r"\s+\d{1,2}$", "", text)

    return text.strip()


# ============================================================
# 5. CHIA TOKEN THANH CAC CHUNK CHONG LAP
# ============================================================

def make_chunks(text, tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    token_ids = encoded["input_ids"]

    # PhoBERT them token dac biet o dau va cuoi
    content_limit = MAX_LENGTH - 2
    step = content_limit - OVERLAP

    if step <= 0:
        raise ValueError(
            "OVERLAP phai nho hon MAX_LENGTH - 2"
        )

    chunks = []

    for start in range(0, len(token_ids), step):
        end = min(start + content_limit, len(token_ids))
        ids = token_ids[start:end]

        if not ids:
            continue

        ids = tokenizer.build_inputs_with_special_tokens(ids)
        chunks.append(ids)

        if end >= len(token_ids):
            break

    return chunks, len(token_ids)


# ============================================================
# 6. GOM CAC TOKEN BIO THANH THUC THE
# ============================================================

def predict_entities(text, tokenizer, model):
    chunks, total_tokens = make_chunks(text, tokenizer)

    all_entities = []
    special_ids = set(tokenizer.all_special_ids)
    id2label = model.config.id2label

    if DEBUG:
        print(f"So token noi dung: {total_tokens}")
        print(f"So chunk xu ly: {len(chunks)}")

    for chunk_idx, chunk_ids in enumerate(chunks, 1):
        input_ids = torch.tensor(
            [chunk_ids],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )[0].tolist()

        ids = input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)

        entities = []
        current = None

        def flush_current():
            nonlocal current

            if current is not None:
                word = current["word"].strip()

                if word:
                    scores = current.pop("scores")
                    current.pop("continues", None)

                    current["word"] = word
                    current["score"] = sum(scores) / len(scores)
                    entities.append(current)

                current = None

        if DEBUG:
            print(f"\n--- CHUNK {chunk_idx}/{len(chunks)} ---")

        for i, (token, pred_id) in enumerate(
            zip(tokens, predictions)
        ):
            if ids[i] in special_ids:
                flush_current()
                continue

            label = id2label.get(
                pred_id,
                id2label.get(str(pred_id), "O"),
            )
            score = probs[i][pred_id].item()

            if DEBUG and label != "O":
                print(
                    f"{token:25s} -> {label:10s} "
                    f"({score:.3f})"
                )

            # PhoBERT fastBPE dung @@ de bao hieu subword
            clean_tok = token.replace("@@", "")

            if label == "O" or "-" not in label:
                flush_current()
                continue

            prefix, entity_type = label.split("-", 1)

            if entity_type not in TARGET_FIELDS:
                flush_current()
                continue

            if prefix == "B" or (
                current is not None
                and current["type"] != entity_type
            ):
                flush_current()

                current = {
                    "type": entity_type,
                    "word": clean_tok,
                    "scores": [score],
                    "continues": token.endswith("@@"),
                }

            elif prefix == "I" and current is not None:
                # Neu subword truoc ket thuc bang @@ thi ghep lien
                if current["continues"]:
                    current["word"] += clean_tok
                else:
                    current["word"] += " " + clean_tok

                current["scores"].append(score)
                current["continues"] = token.endswith("@@")

            elif prefix == "I":
                # I- khong co thuc the truoc do
                current = {
                    "type": entity_type,
                    "word": clean_tok,
                    "scores": [score],
                    "continues": token.endswith("@@"),
                }

            else:
                flush_current()

        flush_current()
        all_entities.extend(entities)

    # Loai cac thuc the trung lap do cac chunk chong lap
    deduplicated = []
    seen = set()

    for entity in all_entities:
        key = (
            entity["type"],
            re.sub(r"\s+", " ", entity["word"]).strip().lower(),
        )

        if key not in seen:
            seen.add(key)
            deduplicated.append(entity)

    return deduplicated, total_tokens, len(chunks)


# ============================================================
# 7. TRICH XUAT VA CHUAN HOA KET QUA
# ============================================================

def process_file(file_path, tokenizer, model):
    full_text = read_ocr_file(file_path)

    if not full_text:
        print(f"File OCR rong: {file_path.name}")
        return

    entities, total_tokens, num_chunks = predict_entities(
        full_text,
        tokenizer,
        model,
    )

    extracted = {
        "STORE": [],
        "DATE": [],
        "TOTAL": [],
        "PROD": [],
    }

    print("\n--- CAC THUC THE SAU KHI GOM ---")

    for ent in entities:
        kind = ent["type"]
        raw = ent["word"]
        score = ent["score"]

        print(
            f"{kind:6s} | {raw} | confidence={score:.3f}"
        )

        if kind == "PROD":
            cleaned = clean_product_name(raw)

            if (
                len(cleaned) > 2
                and re.search(r"[a-zA-ZÀ-ỹ]", cleaned)
            ):
                existing = [
                    item.rsplit(" (", 1)[0]
                    for item in extracted["PROD"]
                ]

                if cleaned not in existing:
                    extracted["PROD"].append(
                        f"{cleaned} ({score:.2f})"
                    )

        elif kind == "STORE":
            cleaned = re.sub(r"\s+", " ", raw).strip()

            if len(cleaned) > 1:
                extracted["STORE"].append(cleaned)

        elif kind in ("DATE", "TOTAL"):
            cleaned = re.sub(r"\s+", " ", raw).strip()

            if cleaned:
                extracted[kind].append(cleaned)

    # Uu tien cac quy tac hau xu ly tu code goc
    final_store = format_store_name(extracted["STORE"])
    final_date = get_ultimate_date(full_text)
    final_total = get_ultimate_total(full_text)

    # Neu regex khong tim thay thi dung ket qua NER lam du phong
    if final_date is None and extracted["DATE"]:
        final_date = extracted["DATE"][0]

    if final_total is None and extracted["TOTAL"]:
        final_total = extracted["TOTAL"][0]

    final_result = {
        "source": file_path.name,
        "STORE": final_store,
        "DATE": final_date or "Khong tim thay",
        "TOTAL": final_total or "Khong tim thay",
        "PROD": extracted["PROD"],
        "entities": entities,
        "raw_text": full_text,
        "total_tokens": total_tokens,
        "num_chunks": num_chunks,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{file_path.stem}.json"
    output_file.write_text(
        json.dumps(
            final_result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n--- KET QUA CUOI ---")
    print("Cua hang :", final_result["STORE"])
    print("Ngay mua :", final_result["DATE"])
    print("Tong tien:", final_result["TOTAL"])
    print(
        "San pham :",
        ", ".join(extracted["PROD"]) or "Khong tim thay",
    )

    if total_tokens > MAX_LENGTH:
        print(
            f"\nHoa don co {total_tokens} token, "
            f"duoc chia thanh {num_chunks} chunk."
        )

    print(f"Da luu ket qua: {output_file}")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_DIR}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc OCR: {INPUT_DIR}"
        )

    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"Khong tim thay file TXT trong {INPUT_DIR}")
        return

    tokenizer, model = load_model()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, file_path in enumerate(txt_files, 1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(txt_files)}] FILE: {file_path.name}")

        try:
            process_file(file_path, tokenizer, model)
        except Exception as error:
            import traceback
            print(f"Loi xu ly {file_path.name}: {error}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
