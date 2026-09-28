import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ Không tìm thấy GEMINI_API_KEY!")

genai.configure(api_key=API_KEY)

INPUT_DIR = "output"
PARSED_DIR = "parsed_info_ai"
os.makedirs(PARSED_DIR, exist_ok=True)

def extract_batch_with_ai(batch_data):
    """
    Nhận vào một mảng chứa thông tin của nhiều file, gửi 1 lần cho AI.
    batch_data có dạng: [{"filename": "test1.txt", "content": "..."}, ...]
    """
    model = genai.GenerativeModel('gemini-3.8-flash')
    
    # 1. Ghép nội dung của nhiều hóa đơn vào một Prompt duy nhất
    prompt = "Bạn là một hệ thống trích xuất dữ liệu hóa đơn. Dưới đây là văn bản OCR của NHIỀU hóa đơn khác nhau:\n\n"
    
    for item in batch_data:
        prompt += f"--- BẮT ĐẦU {item['filename']} ---\n{item['content']}\n--- KẾT THÚC {item['filename']} ---\n\n"
        
    # 2. Định hướng AI trả về một Mảng (List) JSON
    prompt += """
    Hãy trích xuất thông tin cho TỪNG hóa đơn và trả về ĐÚNG MỘT MẢNG (ARRAY) JSON chuẩn.
    Không kèm markdown code block (```json), không giải thích thêm.
    Cấu trúc mong muốn:
    [
      {
        "ten_file": "Tên file tương ứng (ví dụ test1.txt)",
        "cua_hang": "Tên và địa chỉ cửa hàng (nếu có)",
        "ngay_ban": "Ngày tháng năm",
        "tong_tien_phai_tra": "Số tiền tổng cộng (chỉ ghi số)",
        "danh_sach_san_pham": ["tên sản phẩm 1", "tên sản phẩm 2"]
      },
      ...
    ]
    Bỏ qua thông tin rác. Nếu không tìm thấy trường nào, để chuỗi rỗng "".
    """
    
    try:
        response = model.generate_content(prompt)
        # Làm sạch JSON (bỏ markdown nếu có)
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
    except Exception as e:
        print(f"⚠️ Lỗi gọi AI cho cụm file này: {e}")
        return None

def main():
    print("--- HỆ THỐNG TRÍCH XUẤT (RIE): BATCH AI EXTRACTION ---")
    
    if not os.path.exists(INPUT_DIR):
        print(f"⚠️ Chưa tìm thấy thư mục '{INPUT_DIR}'.")
        return
        
    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"⚠️ Không có file text nào trong '{INPUT_DIR}' để xử lý.")
        return
    
    print(f"[*] Tìm thấy {len(txt_files)} file. Đang chuẩn bị xử lý gộp...")
    
    # Đọc trước nội dung của toàn bộ file
    all_data = []
    for filename in txt_files:
        with open(os.path.join(INPUT_DIR, filename), 'r', encoding='utf-8') as f:
            all_data.append({"filename": filename, "content": f.read()})
            
    # Chia nhỏ thành các cụm (Mỗi cụm 5 file)
    BATCH_SIZE = 5
    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i + BATCH_SIZE]
        filenames_in_batch = [item['filename'] for item in batch]
        print(f"\n🚀 Đang gửi cụm {len(batch)} file cho AI: {', '.join(filenames_in_batch)}")
        
        # Gọi AI 1 lần cho cả 5 file
        extracted_list = extract_batch_with_ai(batch)
        
        if extracted_list and isinstance(extracted_list, list):
            # Tách kết quả mảng JSON ra và lưu lại thành từng file riêng lẻ như cũ
            for data in extracted_list:
                original_filename = data.get("ten_file", "unknown.txt")
                
                # Xóa key 'ten_file' trước khi lưu để JSON output giống hệt code cũ của bạn
                if "ten_file" in data:
                    del data["ten_file"]
                    
                out_path = os.path.join(PARSED_DIR, original_filename.replace('.txt', '.json'))
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f"  -> ✅ Đã lưu: {out_path}")
        else:
            print("  -> ⚠️ Trích xuất cụm này thất bại hoặc AI không trả về đúng định dạng mảng.")

if __name__ == "__main__":
    main()