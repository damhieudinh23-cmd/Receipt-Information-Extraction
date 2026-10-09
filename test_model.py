import glob
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "./my_model"
OUTPUT_TXT_DIR = "./output"

print("🔄 Đang tải mô hình PhoBERT NER...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

labels_list = ["O", "B-STORE", "I-STORE", "B-DATE", "I-DATE", "B-TOTAL", "I-TOTAL", "B-PROD", "I-PROD"]
id2label = {i: l for i, l in enumerate(labels_list)}
txt_files = sorted(glob.glob(os.path.join(OUTPUT_TXT_DIR, "*.txt")))[:7]

# ==============================================================================
# 1. CÁC HÀM BỘ LỌC KINH NGHIỆM THỰC TẾ (SIÊU CHUẨN XÁC)
# ==============================================================================

def get_ultimate_total(text):
    """Mẹo: Lấy số tiền có giá trị lớn nhất trong 20 dòng cuối hóa đơn"""
    footer = " ".join(text.split('\n')[-20:])
    # Tìm các chuỗi số tiền tệ bắt đầu từ 1-9 (tránh sđt) vd: 2,442,000 hoặc 81.302
    matches = re.findall(r'\b[1-9]\d{0,2}(?:[.,]\d{3})+\b', footer)
    if not matches: 
        return None
        
    max_val, best_str = -1, ""
    for m in matches:
        # Xóa dấu chấm, phẩy để so sánh độ lớn
        val = int(re.sub(r'[.,]', '', m))
        if val > max_val:
            max_val = val
            best_str = m
    return best_str

def get_ultimate_date(text):
    """Xử lý triệt để lỗi OCR dính chữ (vd: 077/08/2020 -> 07/08/2020)"""
    matches = re.findall(r'\b\d{1,3}[/.-]\d{1,2}[/.-]\d{2,4}\b', text)
    if matches:
        parts = re.split(r'[/.-]', matches[0])
        # Sửa lỗi OCR lặp số (vd: 077 -> lấy 07) bằng cách lấy 2 ký tự đầu
        day = parts[0][:2] if len(parts[0]) > 2 else parts[0].zfill(2)
        month = parts[1][:2] if len(parts[1]) > 2 else parts[1].zfill(2)
        return f"{day}/{month}/{parts[2]}"
    return None

def format_store_name(store_list):
    """Gom các mảnh vỡ của tên cửa hàng do Tokenizer cắt vụn"""
    if not store_list: return "Không tìm thấy"
    raw = "".join([s.split(" (")[0] for s in store_list]).replace(" ", "").lower()
    
    if "vin" in raw or "comerce" in raw: return "Vincommerce"
    if "uni" in raw or "qlo" in raw: return "UNIQLO"
    if "mini" in raw or "anan" in raw: return "MINIMART ANAN"
    return raw.upper()

def clean_product_name(prod_str):
    """Lọc sạch mã vạch, giá tiền và ghép các chữ bị xé lẻ của Sản phẩm"""
    text = prod_str.replace("@@", "").replace("<unk>", "")
    
    # 1. Bỏ mã vạch liền nhau (vd: 08936013233918)
    text = re.sub(r'\b\d{7,15}\b', '', text)
    
    # 2. Bỏ giá tiền chuẩn (vd: 35.000, 149,000)
    text = re.sub(r'\b\d{1,3}[.,]\d{3}\b', '', text)
    
    # 3. [MỚI] Bỏ giá tiền bị VietOCR làm đứt khúc (vd: 34 300, 6 8 600, 26 000)
    text = re.sub(r'\b\d{1,3}\s+\d{3}\b', '', text)
    text = re.sub(r'\b\d\s+\d\s+\d{3}\b', '', text)
    
    # 4. [MỚI] Bỏ các chuỗi mã vạch bị đứt khúc dài ngoẵng (vd: 0 38 9 38 5000 75)
    text = re.sub(r'\b(?:\d+\s+){3,}\d+\b', '', text)
    
    # 5. Bỏ các ký tự rác đặc biệt, chỉ giữ chữ cái, số, %, / và khoảng trắng
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s%/-]', ' ', text)
    
    # 6. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Ghép chữ tiếng Anh đứt khúc (vd: "di el ac" -> "dielac")
    text = re.sub(r'\b([a-zA-Z]{1,2})\s+([a-zA-Z]{1,3})\b', r'\1\2', text)
    text = re.sub(r'\b([a-zA-Z]{1,2})\s+([a-zA-Z]{1,3})\b', r'\1\2', text)
    
    # 8. [MỚI] Dọn dẹp các con số mồ côi (1-2 chữ số) vô nghĩa nằm ở cuối tên sản phẩm
    text = re.sub(r'\s+\d{1,2}$', '', text)
    
    return text.strip()


# ==============================================================================
# 2. CHƯƠNG TRÌNH TRÍCH XUẤT CHÍNH
# ==============================================================================

if not txt_files:
    print(f"❌ Không tìm thấy file .txt nào trong thư mục '{OUTPUT_TXT_DIR}'!")
else:
    print(f"✅ Bắt đầu trích xuất {len(txt_files)} file:\n")

for idx, file_path in enumerate(txt_files, 1):
    file_name = os.path.basename(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
        
    print("==================================================")
    print(f"📄 [{idx}/7] File: {file_name}")
    if not full_text: continue

    # VÌ DATE VÀ TOTAL ĐÃ ĐƯỢC REGEX XỬ LÝ TOÀN BỘ FILE -> Ưu tiên 250 từ đầu cho PhoBERT bắt STORE và PROD
    words = full_text.split()
    smart_text = " ".join(words[:250]) if len(words) > 250 else full_text

    inputs = tokenizer(smart_text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        probabilities = torch.softmax(outputs.logits, dim=2)[0]

    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    entities = []
    current_entity = None

    for token, pred_id, prob in zip(tokens, predictions, probabilities):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]: continue
        label = id2label[pred_id]
        score = prob[pred_id].item()

        if label != "O":
            entity_type = label.split("-")[1]
            clean_tok = token[:-2] if token.endswith("@@") else token

            if label.startswith("B-") or (current_entity and current_entity["type"] != entity_type):
                if current_entity: entities.append(current_entity)
                current_entity = {"type": entity_type, "word": clean_tok, "scores": [score]}
            elif label.startswith("I-") and current_entity and current_entity["type"] == entity_type:
                if current_entity["word"].endswith("@@"):
                    current_entity["word"] = current_entity["word"][:-2] + clean_tok
                else:
                    current_entity["word"] += " " + clean_tok
                current_entity["scores"].append(score)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity: entities.append(current_entity)

    extracted_data = {"STORE": [], "DATE": [], "TOTAL": [], "PROD": []}
    
    for ent in entities:
        ent_type = ent["type"]
        raw_word = ent["word"]
        avg_score = sum(ent["scores"]) / len(ent["scores"])
        
        if ent_type == "PROD":
            # Dọn dẹp rác, mã vạch, giá tiền khỏi tên sản phẩm
            clean_prod = clean_product_name(raw_word)
            # Chỉ lấy các tên sản phẩm hợp lệ
            if len(clean_prod) > 2 and re.search(r'[a-zA-ZÀ-ỹ]', clean_prod):
                if not any(clean_prod in p for p in extracted_data["PROD"]):
                    extracted_data["PROD"].append(f"{clean_prod} ({avg_score:.2f})")
        else:
            # Dọn dẹp cơ bản cho STORE
            clean_w = raw_word.replace("@@", "").replace("<unk>", "").strip()
            clean_w = re.sub(r'\s+', ' ', clean_w)
            if len(clean_w) > 1:
                extracted_data[ent_type].append(f"{clean_w} ({avg_score:.2f})")

    # --- CHỐT HẠ KẾT QUẢ CUỐI CÙNG ---
    final_store = format_store_name(extracted_data["STORE"])
    final_date = get_ultimate_date(full_text)
    final_total = get_ultimate_total(full_text)
    
    print("🎯 KẾT QUẢ TRÍCH XUẤT ĐÃ LÀM SẠCH HOÀN HẢO:")
    print(f"  • Cửa hàng (STORE) : {final_store}")
    print(f"  • Ngày tháng (DATE): {final_date if final_date else 'Không tìm thấy'}")
    print(f"  • Tổng tiền (TOTAL): {final_total if final_total else 'Không tìm thấy'}")
    print(f"  • Sản phẩm (PROD)  : {', '.join(extracted_data['PROD']) if extracted_data['PROD'] else 'Không tìm thấy'}\n")