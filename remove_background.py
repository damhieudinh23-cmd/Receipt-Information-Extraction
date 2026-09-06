import os
from PIL import Image, ImageOps
from rembg import remove

INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEBUG_DIR = "debug"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def main():
    print("--- HỆ THỐNG TRÍCH XUẤT HÓA ĐƠN (RIE): REMOVE BACKGROUND ---")
    files = os.listdir(INPUT_DIR)
    jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(jpg_files) == 0:
        print(f"⚠️ Chưa có ảnh nào trong '{INPUT_DIR}'.")
        return

    for filename in jpg_files:
        print(f"\n[1] Đang xử lý: {filename}...")
        img_path = os.path.join(INPUT_DIR, filename)

        image = Image.open(img_path).convert('RGB')
        
        # Bước 1: Thử xoay theo EXIF nếu có sẵn
        image = ImageOps.exif_transpose(image)

        # Bước 2: Kiểm tra thông minh bằng tỉ lệ khung hình (Aspect Ratio)
        # Nếu chiều rộng > chiều cao (ảnh bị nằm ngang), tự động xoay 270 độ để dựng đứng hóa đơn lên
        width, height = image.size
        if width > height:
            print(f"   [Smart Rotate] Phát hiện ảnh nằm ngang ({width}x{height}) -> Đang tự động xoay thẳng đứng...")
            image = image.rotate(270, expand=True)
        else:
            print(f"   [Smart Rotate] Ảnh đã ở dạng dọc chuẩn ({width}x{height}).")

        # Bước 3: Xóa nền bằng rembg
        print("[3] Đang xóa nền bằng AI (Rembg)...")
        output_image = remove(image)

        # Lưu kết quả vào thư mục debug
        debug_filename = os.path.splitext(filename)[0] + "_nobg.png"
        debug_img_path = os.path.join(DEBUG_DIR, debug_filename)
        output_image.save(debug_img_path)
        print(f"    -> Đã lưu ảnh hoàn chỉnh tại: {debug_img_path}")

if __name__ == "__main__":
    main()