import glob
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "./my_model"
OUTPUT_TXT_DIR = "./output"

print("🔄 Đang tải mô hình PhoBERT NER và Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# Danh sách nhãn chuẩn
labels_list = [
    "O",
    "B-STORE", "I-STORE",
    "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL",
    "B-PROD", "I-PROD"
]
id2label = {i: l for i, l in enumerate(labels_list)}

txt_files = sorted(glob.glob(os.path.join(OUTPUT_TXT_DIR, "*.txt")))[:7]

if not txt_files:
    print(f"❌ Không tìm thấy file .txt nào trong thư mục '{OUTPUT_TXT_DIR}'!")
else:
    print(f"✅ Tìm thấy {len(txt_files)} file .txt đầu tiên. Bắt đầu trích xuất:\n")

for idx, file_path in enumerate(txt_files, 1):
    file_name = os.path.basename(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
        
    print("==================================================")
    print(f"📄 [{idx}/7] File: {file_name}")
    
    if not full_text:
        print("⚠️ File trống, bỏ qua.\n")
        continue

    # 1. Cắt thông minh ở mức từ trước: 100 từ đầu (Header) + 120 từ cuối (Footer)
    words = full_text.split()
    if len(words) > 220:
        smart_text = " ".join(words[:100] + words[-120:])
    else:
        smart_text = full_text

    # 2. Tokenize + Truncate an toàn tuyệt đối với max_length=256
    inputs = tokenizer(
        smart_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        probabilities = torch.softmax(logits, dim=2)[0]

    # 3. Chuyển token IDs ngược lại thành tokens
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 4. Ghép các subword BPE (có '@@') thành từ hoàn chỉnh và gom nhóm thực thể
    entities = []
    current_entity = None

    for token, pred_id, prob in zip(tokens, predictions, probabilities):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        label = id2label[pred_id]
        score = prob[pred_id].item()

        if label != "O":
            entity_type = label.split("-")[1]
            
            # Xử lý ghép subword dính '@@'
            if token.endswith("@@"):
                clean_token = token[:-2]
                is_partial = True
            else:
                clean_token = token
                is_partial = False

            if label.startswith("B-") or (current_entity and current_entity["type"] != entity_type):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": entity_type,
                    "word": clean_token,
                    "scores": [score]
                }
            elif label.startswith("I-") and current_entity and current_entity["type"] == entity_type:
                if current_entity["word"].endswith("@@"):
                    current_entity["word"] = current_entity["word"][:-2] + clean_token
                else:
                    current_entity["word"] += " " + clean_token
                current_entity["scores"].append(score)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # 5. Phân loại kết quả
    extracted_data = {"STORE": [], "DATE": [], "TOTAL": [], "PROD": []}
    for ent in entities:
        ent_type = ent["type"]
        word = ent["word"].replace("@@", "").strip()
        avg_score = sum(ent["scores"]) / len(ent["scores"])
        
        if len(word) > 1 and ent_type in extracted_data:
            extracted_data[ent_type].append(f"{word} ({avg_score:.2f})")

    print("🎯 KẾT QUẢ TRÍCH XUẤT ĐÃ LÀM SẠCH:")
    print(f"  • Cửa hàng (STORE) : {', '.join(extracted_data['STORE']) if extracted_data['STORE'] else 'Không tìm thấy'}")
    print(f"  • Ngày tháng (DATE): {', '.join(extracted_data['DATE']) if extracted_data['DATE'] else 'Không tìm thấy'}")
    print(f"  • Tổng tiền (TOTAL): {', '.join(extracted_data['TOTAL']) if extracted_data['TOTAL'] else 'Không tìm thấy'}")
    print(f"  • Sản phẩm (PROD)  : {', '.join(extracted_data['PROD']) if extracted_data['PROD'] else 'Không tìm thấy'}\n")