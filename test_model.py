import glob
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # Ẩn các cảnh báo/log của transformers

# ============================================================
# NGUỒN DỮ LIỆU CỦA CODE NÀY
#   1) Mô hình : thư mục MODEL_DIR (PhoBERT đã fine-tune cho NER)
#   2) Văn bản : các file .txt trong OUTPUT_TXT_DIR (mỗi file là
#                nội dung chữ của 1 hóa đơn, thường là kết quả OCR)
#   3) Kết quả : TẤT CẢ (STORE, DATE, TOTAL, PROD) đều do mô hình
#                NER dự đoán. Không dùng regex để tìm ngày/tổng tiền,
#                regex chỉ dùng để làm sạch/định dạng lại kết quả.
# ============================================================

MODEL_DIR = "./my_model"
OUTPUT_TXT_DIR = "./output"

MAX_LENGTH = 256
OVERLAP = 48
NUM_FILES = 7

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label

# Chạy trên CPU hoặc GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

txt_files = sorted(
    glob.glob(os.path.join(OUTPUT_TXT_DIR, "*.txt"))
)[:NUM_FILES]


# ============================================================
# 1. CÁC HÀM LÀM SẠCH KẾT QUẢ CỦA MÔ HÌNH
# ============================================================

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


def clean_date(raw):
    # Bỏ khoảng trắng mô hình có thể chèn vào (vd "12 / 05 / 2024")
    text = raw.replace("@@", "").replace(" ", "")

    m = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})', text)

    if not m:
        return text

    day, month, year = m.groups()
    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"


def clean_total(raw):
    # Bỏ khoảng trắng và ký tự lạ, chỉ giữ chữ số và dấu . ,
    text = raw.replace("@@", "").replace(" ", "")
    text = re.sub(r'[^\d.,]', '', text)

    return text.strip(".,")


def clean_product_name(prod_str):
    text = prod_str.replace("@@", "").replace("<unk>", "")

    # Bỏ mã vạch
    text = re.sub(r'\b\d{7,15}\b', ' ', text)

    # Bỏ giá tiền có dấu chấm/phẩy
    text = re.sub(r'\b\d{1,3}[.,]\d{3}\b', ' ', text)

    # Bỏ giá tiền bị tách bằng khoảng trắng
    text = re.sub(r'\b\d{1,3}(?:\s+\d{3})+\b', ' ', text)
    text = re.sub(r'\b\d(?:\s+\d){1,3}\s+\d{3}\b', ' ', text)

    # Bỏ các chuỗi số dài
    text = re.sub(r'\b(?:\d+\s+){3,}\d+\b', ' ', text)

    # Chỉ giữ chữ, số và một số ký tự cần thiết
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s%/-]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    # Xóa số lẻ ở cuối tên sản phẩm
    text = re.sub(r'\s+\d{1,2}$', '', text)

    return text.strip()


def best_by_score(candidates):
    # candidates: list các (giá trị, độ tin cậy)
    # Chọn giá trị có độ tin cậy cao nhất do mô hình dự đoán
    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])[0]


# ============================================================
# 2. CHIA VĂN BẢN THÀNH CÁC ĐOẠN TOKEN CHỒNG LẤP
# ============================================================

def make_chunks(text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )

    token_ids = encoded["input_ids"]

    # PhoBERT cần 2 token đặc biệt ở đầu và cuối
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

        # Thêm token đặc biệt theo đúng tokenizer
        ids = tokenizer.build_inputs_with_special_tokens(ids)

        chunks.append(ids)

        if end >= len(token_ids):
            break

    return chunks


# ============================================================
# 3. DỰ ĐOÁN NHÃN BIO TRÊN CÁC CHUNK
# ============================================================

def predict_entities(text):
    # NGUỒN: mô hình PhoBERT NER (MODEL_DIR) chạy trên văn bản hóa đơn.
    # Mỗi token được gán 1 nhãn BIO, rồi gom lại thành các thực thể
    # (STORE, DATE, TOTAL, PROD) kèm độ tin cậy.
    chunks = make_chunks(text)

    all_entities = []
    special_ids = set(tokenizer.all_special_ids)

    for chunk_ids in chunks:
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

        for i, (token, pred_id) in enumerate(
            zip(tokens, predictions)
        ):
            if ids[i] in special_ids:
                continue

            label = id2label.get(pred_id, str(pred_id))
            score = probs[i][pred_id].item()

            # Bỏ các ký tự phân tách subword nếu có
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
                # Ghép các subtoken theo quy ước tokenizer
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
                # I- không có thực thể trước đó
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

    return all_entities


# ============================================================
# 4. CHẠY CÁC FILE TXT VÀ CHỈ IN KẾT QUẢ CUỐI
# ============================================================

if not txt_files:
    print(f"Khong tim thay file TXT trong thu muc: {OUTPUT_TXT_DIR}")

for idx, file_path in enumerate(txt_files, 1):

    file_name = os.path.basename(file_path)

    # Đọc toàn bộ nội dung 1 hóa đơn từ file .txt (dữ liệu đầu vào)
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    print("\n" + "=" * 70)
    print(f"[{idx}/{len(txt_files)}] FILE: {file_name}")

    if not full_text:
        print("File rong")
        continue

    entities = predict_entities(full_text)

    stores = []          # tên cửa hàng (chuỗi)
    dates = []           # (ngày đã làm sạch, độ tin cậy)
    totals = []          # (tổng tiền đã làm sạch, độ tin cậy)
    products = []        # tên sản phẩm (kèm độ tin cậy)

    for ent in entities:
        kind = ent["type"]
        raw = ent["word"]
        score = ent["score"]

        if kind == "STORE":
            cleaned = re.sub(r'\s+', ' ', raw).strip()

            if len(cleaned) > 1:
                stores.append(cleaned)

        elif kind == "DATE":
            cleaned = clean_date(raw)

            if cleaned:
                dates.append((cleaned, score))

        elif kind == "TOTAL":
            cleaned = clean_total(raw)

            if cleaned:
                totals.append((cleaned, score))

        elif kind == "PROD":
            cleaned = clean_product_name(raw)

            if (
                len(cleaned) > 2
                and re.search(r'[a-zA-ZÀ-ỹ]', cleaned)
            ):
                # Không thêm lại kết quả trùng
                existing = [
                    p.rsplit(" (", 1)[0]
                    for p in products
                ]

                if cleaned not in existing:
                    products.append(f"{cleaned} ({score:.2f})")

    # Tổng hợp kết quả cuối (tất cả đều từ mô hình NER):
    #   - Cửa hàng : các thực thể STORE (+ chuẩn hóa tên)
    #   - Ngày mua : thực thể DATE có độ tin cậy cao nhất
    #   - Tổng tiền: thực thể TOTAL có độ tin cậy cao nhất
    #   - Sản phẩm : các thực thể PROD (+ làm sạch)
    final_store = format_store_name(stores)
    final_date = best_by_score(dates)
    final_total = best_by_score(totals)

    print("Cua hang :", final_store)
    print("Ngay mua :", final_date or "Khong tim thay")
    print("Tong tien:", final_total or "Khong tim thay")
    print(
        "San pham :",
        ", ".join(products) or "Khong tim thay"
    )