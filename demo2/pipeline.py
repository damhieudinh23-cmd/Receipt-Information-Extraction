from PIL import Image, ImageOps, ImageDraw
from config import ORIENTED_DIR
from ocr_engine import evaluate_orientation, extract_words_and_boxes
from postprocess import (
    group_lines,
    extract_store,
    extract_date,
    extract_total
)

LABEL_COLORS = {
    "SELLER": "#E63946",      # Đỏ
    "TIMESTAMP": "#2A9D8F",   # Xanh ngọc
    "TOTAL_COST": "#FF9F1C",  # Cam
    "OTHER": "#8D99AE",       # Xám
}


def draw_predictions_on_image(image, records):
    """Vẽ bounding box các nhãn quan trọng lên ảnh để hiển thị trên app demo."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for r in records:
        label = str(r.get("label", "OTHER")).upper()
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]

        color = "#8D99AE"
        for key, hex_color in LABEL_COLORS.items():
            if key in label:
                color = hex_color
                break

        if "OTHER" in label or label == "O":
            draw.rectangle([x, y, x + w, y + h], outline="#CCCCCC", width=1)
        else:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x, max(0, y - 14)), label, fill=color)

    return annotated


def process_receipt(
    image_source,
    predictor,
    image_name="uploaded.jpg",
    save_oriented=True,
    verbose=True
):
    """
    Xử lý 1 ảnh hóa đơn (nhận đường dẫn Path hoặc đối tượng PIL.Image).
    """
    if isinstance(image_source, Image.Image):
        image = ImageOps.exif_transpose(image_source).convert("RGB")
    else:
        with Image.open(image_source) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        image_name = image_source.name

    if verbose:
        print("Chon huong anh:")
    oriented_image, best_angle = evaluate_orientation(image, verbose=verbose)

    if save_oriented:
        stem = image_name.rsplit(".", 1)[0]
        oriented_image.save(ORIENTED_DIR / f"{stem}_oriented.png")

    records = extract_words_and_boxes(oriented_image)
    records = predictor.predict_words(oriented_image, records)
    lines = group_lines(records)

    store = extract_store(records, lines, oriented_image.height)
    date = extract_date(lines)
    total, total_raw = extract_total(records, lines)

    annotated_image = draw_predictions_on_image(oriented_image, records)

    result = {
        "image": image_name,
        "angle": best_angle,
        "STORE": store,
        "DATE": date,
        "TOTAL": total,
        "TOTAL_RAW": total_raw
    }

    return {
        "oriented_image": oriented_image,
        "annotated_image": annotated_image,
        "records": records,
        "lines": lines,
        "result": result
    }