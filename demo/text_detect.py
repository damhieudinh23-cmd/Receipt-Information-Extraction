
from pathlib import Path

import cv2
from craft_text_detector import Craft


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "demo_output"
OUTPUT_DIR = BASE_DIR / "detected"


def detect_text(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc ảnh gốc để cắt các vùng chữ
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    height, width = image.shape[:2]

    craft = Craft(
        output_dir=str(output_dir),
        crop_type="poly",
        cuda=False
    )

    try:
        # Phát hiện vùng chữ
        result = craft.detect_text(str(image_path))
        boxes = result.get("boxes")

        if boxes is None or len(boxes) == 0:
            print(f"Không phát hiện được chữ: {image_path.name}")
            return []

        # Tạo ảnh minh họa các vùng chữ
        preview = image.copy()
        crop_paths = []

        # Sắp xếp gần theo thứ tự đọc: từ trên xuống dưới,
        # trong mỗi vùng ngang thì từ trái sang phải
        boxes = sorted(
            boxes,
            key=lambda box: (
                float(box[:, 1].mean()),
                float(box[:, 0].mean())
            )
        )

        for index, box in enumerate(boxes):
            # Lấy hình chữ nhật bao quanh vùng chữ
            x1 = max(0, int(box[:, 0].min()))
            y1 = max(0, int(box[:, 1].min()))
            x2 = min(width, int(box[:, 0].max()) + 1)
            y2 = min(height, int(box[:, 1].max()) + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_path = output_dir / f"crop_{index:04d}.png"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

            # Vẽ bounding box lên ảnh xem trước
            points = box.astype("int32").reshape((-1, 1, 2))
            cv2.polylines(
                preview,
                [points],
                True,
                (0, 255, 0),
                2
            )

        preview_path = output_dir / "detection_preview.jpg"
        cv2.imwrite(str(preview_path), preview)

        print(f"Ảnh: {image_path.name}")
        print(f"Số vùng chữ: {len(crop_paths)}")
        print(f"Ảnh crop lưu tại: {output_dir}")
        print(f"Ảnh xem trước: {preview_path}")

        return crop_paths

    finally:
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*_processed.png"))

    if not image_paths:
        print(f"Không tìm thấy ảnh trong {INPUT_DIR}")
        return

    for image_path in image_paths:
        # Mỗi hóa đơn có một thư mục riêng
        invoice_dir = OUTPUT_DIR / image_path.stem

        try:
            detect_text(image_path, invoice_dir)
        except Exception as e:
            import traceback
            print(f"Lỗi {image_path.name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()