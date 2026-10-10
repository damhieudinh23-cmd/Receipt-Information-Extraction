
import glob
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "./my_model"
OUTPUT_TXT_DIR = "./output"

MAX_LENGTH = 256
OVERLAP = 48
NUM_FILES = 7
DEBUG = True

print("Dang tai mo hinh PhoBERT NER...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label

# Chay tren CPU hoac GPU neu co
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

txt_files = sorted(
    glob.glob(os.path.join(OUTPUT_TXT_DIR, "*.txt"))
)[:NUM_FILES]


# ============================================================
# 1. CAC HAM HAU XU LY
# ============================================================

def get_ultimate_total(text):
    footer = " ".join(text.splitlines()[-20:])

    matches = re.findall(
        r'\b[1-9]\d{0,2}(?:[.,]\d{3})+\b',
        footer
    )

    if not matches:
        return None

    return max(
        matches,
        key=lambda x: int(re.sub(r'[.,]', '', x))
    )


def get_ultimate_date(text):
    matches = re.findall(
        r'(?<!\d)\d{1,3}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)',
        text
    )

    if not matches:
        return None

    parts = re.split(r'[/.-]', matches[0])

    day = parts[0][:2] if len(parts[0]) > 2 else parts[0].zfill(2)
    month = parts[1][:2] if len(parts[1]) > 2 else parts[1].zfill(2)

    return f"{day}/{month}/{parts[2]}"


def format_store_name(store_list):
    if not store_list:
        return "Khong tim thay"

    raw = "".join(store_list).replace(" ", "").lower()

    if "vin" in raw or "comerce" in raw:
        return "Vincommerce"

    if "uni" in raw or "qlo" in raw:
        return "UNIQLO"

    if "mini" in raw or "anan" in raw:
        return "MINIMART ANAN"

    return raw.upper()


def clean_product_name(prod_str):
    text = prod_str.replace("@@", "").replace("<unk>", "")

    # Bo ma vach
    text = re.sub(r'\b\d{7,15}\b', ' ', text)

    # Bo gia tien co dau cham/phay
    text = re.sub(r'\b\d{1,3}[.,]\d{3}\b', ' ', text)

    # Bo gia tien bi tach bang khoang trang
    text = re.sub(r'\b\d{1,3}(?:\s+\d{3})+\b', ' ', text)
    text = re.sub(r'\b\d(?:\s+\d){1,3}\s+\d{3}\b', ' ', text)

    # Bo cac chuoi so dai
    text = re.sub(r'\b(?:\d+\s+){3,}\d+\b', ' ', text)

    # Chi giu chu, so va mot so ky tu can thiet
    text = re.sub(
        r'[^a-zA-ZÀ-ỹ0-9\s%/-]',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    # Xoa so le o cuoi ten san pham
    text = re.sub(r'\s+\d{1,2}$', '', text)

    return text.strip()


# ============================================================
# 2. CHIA VAN BAN THANH CAC DOAN TOKEN CHONG LAP
# ============================================================

def make_chunks(text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )

    token_ids = encoded["input_ids"]

    # PhoBERT can cho 2 token dac biet o dau va cuoi
    content_limit = MAX_LENGTH - 2
    step = content_limit - OVERLAP

    chunks = []

    if step <= 0:
        raise ValueError("OVERLAP phai nho hon MAX_LENGTH - 2")

    for start in range(0, len(token_ids), step):
        end = min(start + content_limit, len(token_ids))
        ids = token_ids[start:end]

        if not ids:
            continue

        # Them token dac biet theo dung tokenizer
        ids = tokenizer.build_inputs_with_special_tokens(ids)

        chunks.append(ids)

        if end >= len(token_ids):
            break

    return chunks, len(token_ids)


# ============================================================
# 3. DU DOAN NHAN BIO TREN CAC CHUNK
# ============================================================

def predict_entities(text):
    chunks, total_tokens = make_chunks(text)

    all_entities = []
    special_ids = set(tokenizer.all_special_ids)

    if DEBUG:
        print(f"So token noi dung toan van ban: {total_tokens}")
        print(f"So doan xu ly: {len(chunks)}")

    for chunk_idx, chunk_ids in enumerate(chunks, 1):
        input_ids = torch.tensor(
            [chunk_ids],
            dtype=torch.long,
            device=device
        )

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predictions = torch.argmax(
                outputs.logits,
                dim=-1
            )[0].tolist()

        ids = input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)

        entities = []
        current = None

        if DEBUG:
            print(
                f"\n--- CHUNK {chunk_idx}/{len(chunks)} ---"
            )

        for i, (token, pred_id) in enumerate(
            zip(tokens, predictions)
        ):
            if ids[i] in special_ids:
                continue

            label = id2label.get(pred_id, str(pred_id))
            score = probs[i][pred_id].item()

            if DEBUG and label != "O":
                print(
                    f"{token:25s} -> {label:10s} "
                    f"({score:.3f})"
                )

            # Bo cac ky tu phan tach subword neu co
            clean_tok = token.replace("@@", "")

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            if not label.startswith(("B-", "I-")):
                if current:
                    entities.append(current)
                    current = None
                continue

            entity_type = label.split("-", 1)[1]

            if label.startswith("B-") or (
                current and current["type"] != entity_type
            ):
                if current:
                    entities.append(current)

                current = {
                    "type": entity_type,
                    "word": clean_tok,
                    "scores": [score]
                }

            elif label.startswith("I-") and current:
                # Ghép cac subtoken theo quy uoc tokenizer
                if token.startswith("##"):
                    current["word"] += token[2:]
                elif current["word"].endswith("@@"):
                    current["word"] = (
                        current["word"][:-2] + clean_tok
                    )
                else:
                    current["word"] += " " + clean_tok

                current["scores"].append(score)

            else:
                # I- khong co thuc the truoc do
                current = {
                    "type": entity_type,
                    "word": clean_tok,
                    "scores": [score]
                }

        if current:
            entities.append(current)

        for ent in entities:
            ent["word"] = ent["word"].strip()
            ent["score"] = (
                sum(ent["scores"]) / len(ent["scores"])
            )

        all_entities.extend(entities)

    return all_entities, total_tokens, len(chunks)


# ============================================================
# 4. CHAY THU CAC FILE TXT
# ============================================================

if not txt_files:
    print(f"Khong tim thay file TXT trong thu muc: {OUTPUT_TXT_DIR}")

for idx, file_path in enumerate(txt_files, 1):

    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    print("\n" + "=" * 70)
    print(f"[{idx}/{len(txt_files)}] FILE: {file_name}")

    if not full_text:
        print("File rong")
        continue

    entities, total_tokens, num_chunks = predict_entities(full_text)

    extracted = {
        "STORE": [],
        "DATE": [],
        "TOTAL": [],
        "PROD": []
    }

    print("\n--- CAC THUC THE SAU KHI GOM ---")

    for ent in entities:
        kind = ent["type"]
        raw = ent["word"]
        score = ent["score"]

        print(
            f'{kind:6s} | {raw} | confidence={score:.3f}'
        )

        if kind not in extracted:
            continue

        if kind == "PROD":
            cleaned = clean_product_name(raw)

            if (
                len(cleaned) > 2
                and re.search(r'[a-zA-ZÀ-ỹ]', cleaned)
            ):
                # Khong them lai ket qua trung
                existing = [
                    p.rsplit(" (", 1)[0]
                    for p in extracted["PROD"]
                ]

                if cleaned not in existing:
                    extracted["PROD"].append(
                        f"{cleaned} ({score:.2f})"
                    )

        elif kind == "STORE":
            cleaned = re.sub(r'\s+', ' ', raw).strip()

            if len(cleaned) > 1:
                extracted["STORE"].append(cleaned)

        elif kind in ("DATE", "TOTAL"):
            cleaned = re.sub(r'\s+', ' ', raw).strip()

            if len(cleaned) > 0:
                extracted[kind].append(cleaned)

    # DATE va TOTAL van duoc uu tien lay tu regex toan van ban
    final_store = format_store_name(extracted["STORE"])
    final_date = get_ultimate_date(full_text)
    final_total = get_ultimate_total(full_text)

    print("\n--- KET QUA CUOI ---")
    print("Cua hang :", final_store)
    print("Ngay mua :", final_date or "Khong tim thay")
    print("Tong tien:", final_total or "Khong tim thay")
    print(
        "San pham :",
        ", ".join(extracted["PROD"]) or "Khong tim thay"
    )

    if total_tokens > MAX_LENGTH:
        print(
            f"\nThong tin: hoa don co {total_tokens} token, "
            f"duoc chia thanh {num_chunks} doan de xu ly."
        )

