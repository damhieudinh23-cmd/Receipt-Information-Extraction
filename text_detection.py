import os
import cv2
import numpy as np
from PIL import Image
from craft_text_detector import Craft

DEBUG_DIR = "debug"
DETECTED_DIR = "detected"

os.makedirs(DETECTED_DIR, exist_ok=True)

def detect_text_craft(image_path, output_dir):
    print(f"\n[+] Đang chạy CRAFT Text Detection cho: {os.path.basename(image_path)}")
    
    # Chuyển ảnh sang RGB chuẩn
    img_pil = Image.open(image_path).convert('RGB')
    temp_rgb_path = os.path.join(output_dir, "temp_input.jpg")
    img_pil.save(temp_rgb_path)
    
    try:
        # Khởi tạo mô hình CRAFT
        craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)
        
        # Chạy dự đoán
        prediction_result = craft.detect_text(temp_rgb_path)
        
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()
        
        print(f" -> ✅ Thành công! Kết quả lưu tại: '{output_dir}'")
        return prediction_result

    except ValueError as e:
        # Bắt lỗi không tương thích mảng của craft-text-detector
        print(f" ⚠️ CRAFT gặp lỗi định dạng mảng (Inhomogeneous shape), đang chuyển sang chế độ fallback...")
        
        # Chạy phát hiện khung chữ ở dạng rect (chữ nhật) đơn giản để không gãy pipeline
        craft = Craft(output_dir=output_dir, crop_type="rect", cuda=False)
        prediction_result = craft.detect_text(temp_rgb_path)
        
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()
        
        print(f" -> ✅ Thành công (Chế độ Rect)! Kết quả lưu tại: '{output_dir}'")
        return prediction_result

    finally:
        if os.path.exists(temp_rgb_path):
            os.remove(temp_rgb_path)

def process_detection_pipeline():
    if not os.path.exists(DEBUG_DIR):
        print(f"⚠️ Chưa tìm thấy thư mục '{DEBUG_DIR}'. Hãy chạy bước xóa nền (Rembg) trước!")
        return

    nobg_files = [f for f in os.listdir(DEBUG_DIR) if f.endswith("_nobg.png")]

    if len(nobg_files) == 0:
        print(f"⚠️ Không tìm thấy ảnh '_nobg.png' nào trong '{DEBUG_DIR}' để detect chữ.")
        return

    print(f"Tìm thấy {len(nobg_files)} ảnh cần xử lý trong '{DEBUG_DIR}'.")

    for filename in nobg_files:
        img_path = os.path.join(DEBUG_DIR, filename)
        image_out_dir = os.path.join(DETECTED_DIR, filename.replace("_nobg.png", ""))
        os.makedirs(image_out_dir, exist_ok=True)
        
        detect_text_craft(img_path, image_out_dir)

if __name__ == "__main__":
    process_detection_pipeline()