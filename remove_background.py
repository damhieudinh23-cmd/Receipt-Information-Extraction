import os
from PIL import Image
from rembg import remove

# 1. Định nghĩa thư mục
INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEBUG_DIR = "debug" # Thư mục mới để lưu ảnh xem thử

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True) # Tự động tạo thư mục debug

def main():
    print("--- HỆ THỐNG TRÍCH XUẤT HÓA ĐƠN (RIE) KHỞI ĐỘNG ---")
    
    files = os.listdir(INPUT_DIR)
    jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if len(jpg_files) == 0:
        print(f"Chưa có ảnh nào! Bạn hãy copy ít nhất một ảnh hóa đơn (.jpg) vào thư mục '{INPUT_DIR}' nhé.")
        return

    # 2. Xử lý từng ảnh
    for filename in jpg_files:
        print(f"\n[1] Đang đọc ảnh: {filename}...")
        img_path = os.path.join(INPUT_DIR, filename)
        
        # --- BƯỚC 1: XÓA NỀN (BACKGROUND REMOVAL - REMBG) ---
        print("[2] Đang xóa nền bằng AI (Rembg)... (Có thể mất vài giây cho lần chạy đầu tiên)")
        
        # Mở ảnh gốc bằng thư viện PIL
        input_image = Image.open(img_path)
        
        # Gọi AI của Rembg để xóa nền
        output_image = remove(input_image)
        
        # Lưu ảnh đã xóa nền dưới dạng .png (để hỗ trợ nền trong suốt) vào thư mục debug
        debug_filename = filename.replace(".jpg", "_nobg.png").replace(".jpeg", "_nobg.png")
        debug_img_path = os.path.join(DEBUG_DIR, debug_filename)
        output_image.save(debug_img_path)
        print(f"    -> Đã lưu ảnh tách nền thành công tại: {debug_img_path}")
        
        # --- BƯỚC TIẾP THEO SẼ LÀ ĐỌC CHỮ (Sẽ code sau) ---
        print("[3] Đang xuất file kết quả text...")
        extracted_text = f"Tên file gốc: {filename}\nTrạng thái: Đã tách nền thành công! Hãy xem ảnh trong thư mục debug."
        
        out_filename = filename.replace(".jpg", ".txt").replace(".jpeg", ".txt")
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
            
        print(f"-> HOÀN THÀNH! Đã lưu thông tin tại: {out_path}")

if __name__ == "__main__":
    main()