import streamlit as st
import pandas as pd
from PIL import Image

from config import INPUT_DIR, OUTPUT_DIR, IMAGE_EXTENSIONS, DEVICE
from model_engine import LayoutLMv3Predictor
from pipeline import process_receipt

st.set_page_config(
    page_title="Demo Trích Xuất Hóa Đơn - LayoutLMv3",
    layout="wide"
)


@st.cache_resource
def get_predictor():
    """Load model 1 lần duy nhất và giữ trong RAM khi chạy web."""
    return LayoutLMv3Predictor()


def main():
    st.title("🧾 Hệ Thống Trích Xuất Thông Tin Hóa Đơn (LayoutLMv3 + OCR)")
    st.caption(f"Thiết bị đang chạy: **{DEVICE}** | Thư mục đầu vào: `{INPUT_DIR}`")

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Không thể tải model: {e}")
        return

    tab_single, tab_batch = st.tabs([
        "Demo 1 Ảnh Hóa Đơn",
        "Chạy Hàng Loạt (Thư mục demo_input)"
    ])

    # ==========================================
    # TAB 1: XỬ LÝ 1 ẢNH ĐƠN LẺ
    # ==========================================
    with tab_single:
        col_ctrl1, col_ctrl2 = st.columns([1, 1])

        with col_ctrl1:
            uploaded_file = st.file_uploader(
                "Tải ảnh hóa đơn từ máy tính",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )

        with col_ctrl2:
            existing_images = sorted(
                p.name for p in INPUT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            selected_sample = st.selectbox(
                "Hoặc chọn ảnh có sẵn trong demo_input:",
                options=["-- Không chọn --"] + existing_images
            )

        target_image = None
        target_name = None

        if uploaded_file is not None:
            target_image = Image.open(uploaded_file)
            target_name = uploaded_file.name
        elif selected_sample != "-- Không chọn --":
            target_image = INPUT_DIR / selected_sample
            target_name = selected_sample

        if target_image is not None:
            if st.button("Chạy trích xuất", type="primary"):
                with st.spinner("Đang xoay ảnh, nhận diện OCR và dự đoán LayoutLMv3..."):
                    output = process_receipt(
                        target_image,
                        predictor,
                        image_name=target_name,
                        verbose=False
                    )

                res = output["result"]

                # Hiển thị thẻ kết quả nổi bật
                st.subheader(" Kết quả trích xuất")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(" Cửa hàng (STORE)", res["STORE"] or "Không rõ")
                m2.metric(" Ngày mua (DATE)", res["DATE"] or "Không rõ")
                m3.metric(
                    "💰 Tổng tiền (TOTAL)",
                    f"{res['TOTAL']:,} đ" if res["TOTAL"] else "Không rõ"
                )
                m4.metric("Góc xoay tự động", f"{res['angle']}°")

                st.divider()

                # Hiển thị ảnh đối chiếu
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown("**Ảnh sau khi tự động xoay hướng**")
                    st.image(output["oriented_image"], use_container_width=True)
                with img_col2:
                    st.markdown("**Ảnh gán nhãn Bounding Box (LayoutLMv3)**")
                    st.image(output["annotated_image"], use_container_width=True)

                # Hiển thị bảng chi tiết Token OCR
                with st.expander("Xem bảng chi tiết các từ (OCR Tokens & Nhãn dự đoán)"):
                    df_tokens = pd.DataFrame([
                        {
                            "Từ (Word)": r["word"],
                            "Nhãn (Label)": r["label"],
                            "Độ tin cậy Model": round(r["model_confidence"], 4),
                            "Độ tin cậy OCR": round(r["confidence_ocr"], 1),
                            "Tọa độ [x0, y0, x1, y1]": str(r["box"])
                        }
                        for r in output["records"]
                    ])
                    st.dataframe(df_tokens, use_container_width=True)

    # ==========================================
    # TAB 2: CHẠY HÀNG LOẠT TRONG DEMO_INPUT
    # ==========================================
    with tab_batch:
        st.write(f"Tìm thấy **{len(existing_images)}** ảnh trong thư mục `{INPUT_DIR}`.")

        if st.button("Chạy toàn bộ ảnh trong demo_input"):
            if not existing_images:
                st.warning("Thư mục demo_input đang trống!")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                batch_results = []

                for idx, img_name in enumerate(existing_images, 1):
                    status_text.text(f"Đang xử lý [{idx}/{len(existing_images)}]: {img_name}...")
                    img_path = INPUT_DIR / img_name
                    try:
                        out = process_receipt(img_path, predictor, verbose=False)
                        batch_results.append(out["result"])
                    except Exception as e:
                        batch_results.append({
                            "image": img_name,
                            "angle": 0,
                            "STORE": f"Lỗi: {e}",
                            "DATE": None,
                            "TOTAL": None,
                            "TOTAL_RAW": None
                        })
                    progress.progress(idx / len(existing_images))

                status_text.success("Đã xử lý hoàn tất toàn bộ ảnh!")
                df_results = pd.DataFrame(batch_results)
                st.dataframe(df_results, use_container_width=True)

                csv_bytes = df_results.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="Tải xuống kết quả CSV",
                    data=csv_bytes,
                    file_name="extracted_results_demo.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()