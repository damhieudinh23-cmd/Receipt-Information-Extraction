import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import remove
import pytesseract

# Đường dẫn Tesseract trên Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEBUG_DIR = "debug"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def evaluate_orientation(pil_img):
    best_image = pil_img
    max_score = -1
    best_angle = 0

    angles = [0, 90, 180, 270]

    for angle in angles:
        if angle == 0:
            rotated = pil_img
        else:
            rotated = pil_img.rotate(360 - angle, expand=True)

        try:
            cv_img = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # ÉP Tesseract CHỈ đọc theo chiều ngang (--psm 6)
            data = pytesseract.image_to_data(gray, config='--psm 6', output_type=pytesseract.Output.DICT)
            
            valid_words = 0
            sum_conf = 0
            
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Điều kiện khắt khe hơn: Độ tự tin > 50, độ dài > 2
                if conf > 50 and len(text) > 2 and any(c.isalnum() for c in text):
                    valid_words += 1
                    sum_conf += conf
            
            score = sum_conf

            print(f"      + Thử góc {angle}°: Điểm = {score} (Số từ HỢP LỆ: {valid_words})")

            if score > max_score:
                max_score = score
                best_image = rotated
                best_angle = angle

        except Exception as e:
            print(f"      + Thử góc {angle}°: Lỗi quét chữ -> {e}")

    print(f"   => [Smart OCR] CHỐT GÓC CHUẨN: {best_angle}°")
    return best_image

def main():
    print("--- HỆ THỐNG TRÍCH XUẤT HÓA ĐƠN (RIE): PREPROCESSING ---")
    files = os.listdir(INPUT_DIR)
    jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(jpg_files) == 0:
        print(f"⚠️ Chưa có ảnh nào trong '{INPUT_DIR}'.")
        return

    for filename in jpg_files:
        print(f"\n[1] Đang xử lý: {filename}...")
        img_path = os.path.join(INPUT_DIR, filename)

        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)

        print("   [2] Đang quét 4 hướng để tìm chiều chữ chuẩn...")
        image = evaluate_orientation(image)

        print("   [3] Đang xóa nền bằng AI (Rembg)...")
        output_image = remove(image)

        debug_filename = os.path.splitext(filename)[0] + "_nobg.png"
        debug_img_path = os.path.join(DEBUG_DIR, debug_filename)
        output_image.save(debug_img_path)
        print(f"    -> Đã lưu ảnh hoàn chỉnh tại: {debug_img_path}")

if __name__ == "__main__":
    main()