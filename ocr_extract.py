import os
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# --- BẮT ĐẦU ĐOẠN CODE FIX LỖI SSL ---
import requests
import urllib3
# Tắt cảnh báo dòng chữ đỏ phiền phức trên terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ép hệ thống mạng bỏ qua khâu kiểm tra chứng chỉ hết hạn
original_get = requests.get
def unverified_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = unverified_get
# --- KẾT THÚC ĐOẠN CODE FIX LỖI SSL ---

DETECTED_DIR = "detected"
OUTPUT_DIR = "output"

# ... (Giữ nguyên toàn bộ phần code từ os.makedirs trở xuống của bạn) ...

os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_vietocr():
    # Sử dụng mô hình 'vgg_transformer' cho độ chính xác cao nhất với Tiếng Việt
    config = Cfg.load_config_from_name('vgg_transformer')
    
    # Thiết lập chạy trên CPU để tránh lỗi nếu máy không có card đồ họa NVIDIA
    config['device'] = 'cpu' 
    
    predictor = Predictor(config)
    return predictor

def main():
    print("--- HỆ THỐNG TRÍCH XUẤT HÓA ĐƠN (RIE): TEXT RECOGNITION (VIETOCR) ---")
    
    if not os.path.exists(DETECTED_DIR):
        print(f" Chưa tìm thấy thư mục '{DETECTED_DIR}'. Hãy chạy file 'text_detection.py' trước!")
        return

    subfolders = [f for f in os.listdir(DETECTED_DIR) if os.path.isdir(os.path.join(DETECTED_DIR, f))]

    if len(subfolders) == 0:
        print(f" Chưa tìm thấy thư mục con nào trong '{DETECTED_DIR}'.")
        return

    # Khởi tạo AI (lần đầu tiên chạy sẽ mất chút thời gian để máy tự động tải weights)
    print("\n[*] Đang khởi tạo mô hình VietOCR...")
    predictor = setup_vietocr()

    for folder_name in subfolders:
        folder_path = os.path.join(DETECTED_DIR, folder_name)
        print(f"\n[1] Đang đọc các khung chữ từ hóa đơn: {folder_name}...")

        # Dò tìm tất cả các file crop
        crop_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "crop" in f.lower():
                    crop_files.append(os.path.join(root, f))
        
        # Hàm sắp xếp file theo số thứ tự
        def extract_number(filepath):
            filename = os.path.basename(filepath)
            nums = ''.join(filter(str.isdigit, filename))
            return int(nums) if nums else 0
            
        crop_files.sort(key=extract_number)

        if not crop_files:
            print(f"    Không tìm thấy ảnh crop nào trong {folder_path}")
            continue

        full_extracted_text = []

        print(f"   [2] Bắt đầu quét VietOCR trên {len(crop_files)} mảnh cắt...")
        for crop_path in crop_files:
            try:
                crop_img = Image.open(crop_path).convert('RGB')
                
                # Dùng VietOCR để đọc chữ thay vì Tesseract
                text = predictor.predict(crop_img).strip()
                
                if text:
                    full_extracted_text.append(text)
            except Exception as e:
                print(f"    ⚠️ Lỗi đọc ảnh {os.path.basename(crop_path)}: {e}")

        # Gộp toàn bộ văn bản lại và xuất ra file
        out_filename = f"{folder_name}.txt"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_extracted_text))
            
        print(f"     Đã xuất thành công kết quả text tại: {out_path}")

    print("\n -> HOÀN THÀNH TOÀN BỘ PIPELINE RIE!")

if __name__ == "__main__":
    main()