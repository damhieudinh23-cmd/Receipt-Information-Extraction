import sys
import subprocess
from datetime import datetime
import csv
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import pandas as pd
from PIL import Image

from config import INPUT_DIR, OUTPUT_DIR, IMAGE_EXTENSIONS, DEVICE
from model_engine import LayoutLMv3Predictor
from pipeline import process_receipt

HISTORY_CSV = OUTPUT_DIR / "saved_receipts_history.csv"
HISTORY_FIELDS = [
    "saved_at",
    "image",
    "STORE",
    "DATE",
    "TOTAL",
    "TOTAL_RAW",
    "angle"
]


@st.cache_resource
def get_predictor():
    return LayoutLMv3Predictor()


def load_history():
    if not HISTORY_CSV.exists():
        return []
    with open(HISTORY_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_receipt_to_history(record):
    file_exists = HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def rewrite_history(records):
    if not records:
        if HISTORY_CSV.exists():
            HISTORY_CSV.unlink()
        return
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def clear_history():
    if HISTORY_CSV.exists():
        HISTORY_CSV.unlink()


def main():
    st.set_page_config(
        page_title="Demo Trích Xuất Hóa Đơn - LayoutLMv3",
        layout="wide"
    )

    st.title(" Hệ Thống Trích Xuất Thông Tin Hóa Đơn (LayoutLMv3 + OCR)")
    st.caption(f"Thiết bị đang chạy: **{DEVICE}** | Thư mục đầu vào: `{INPUT_DIR}`")

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Không thể tải model: {e}")
        return

    if "last_single_output" not in st.session_state:
        st.session_state["last_single_output"] = None
    if "save_status" not in st.session_state:
        st.session_state["save_status"] = None

    existing_images = sorted(
        p.name for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    history_data = load_history()

    tab_single, tab_batch, tab_history = st.tabs([
        " Demo 1 Ảnh Hóa Đơn",
        " Chạy Hàng Loạt (Thư mục demo_input)",
        f" Lịch Sử Hóa Đơn ({len(history_data)})"
    ])

    with tab_single:
        col_ctrl1, col_ctrl2 = st.columns([1, 1])

        with col_ctrl1:
            uploaded_file = st.file_uploader(
                "Tải ảnh hóa đơn từ máy tính",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )

        with col_ctrl2:
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
            if st.button(" Chạy trích xuất", type="primary"):
                with st.spinner("Đang xoay ảnh, nhận diện OCR và dự đoán LayoutLMv3..."):
                    output = process_receipt(
                        target_image,
                        predictor,
                        image_name=target_name,
                        verbose=False
                    )
                    st.session_state["last_single_output"] = output
                    st.session_state["save_status"] = "ask"

        if st.session_state["last_single_output"] is not None:
            output = st.session_state["last_single_output"]
            res = output["result"]

            st.subheader(" Kết quả trích xuất")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(" Cửa hàng (STORE)", res["STORE"] or "Không rõ")
            m2.metric(" Ngày mua (DATE)", res["DATE"] or "Không rõ")
            m3.metric(
                " Tổng tiền (TOTAL)",
                f"{res['TOTAL']:,} đ" if res["TOTAL"] else "Không rõ"
            )
            m4.metric(" Góc xoay tự động", f"{res['angle']}°")

            if st.session_state["save_status"] == "ask":
                st.info(" **Bạn có muốn lưu thông tin của hóa đơn này vào lịch sử không?** (Có thể chỉnh sửa lại thông tin bên dưới trước khi lưu)")
                with st.container(border=True):
                    edit_c1, edit_c2, edit_c3 = st.columns(3)
                    with edit_c1:
                        store_val = st.text_input(
                            "Tên cửa hàng (STORE)",
                            value=res["STORE"] or "",
                            key="edit_store"
                        )
                    with edit_c2:
                        date_val = st.text_input(
                            "Ngày mua (DATE)",
                            value=res["DATE"] or "",
                            key="edit_date"
                        )
                    with edit_c3:
                        total_val = st.number_input(
                            "Tổng tiền (TOTAL)",
                            min_value=0,
                            value=int(res["TOTAL"]) if res["TOTAL"] else 0,
                            step=1000,
                            key="edit_total"
                        )

                    btn_c1, btn_c2, _ = st.columns([1, 1, 3])
                    with btn_c1:
                        if st.button(" Có, lưu hóa đơn", type="primary", use_container_width=True):
                            save_receipt_to_history({
                                "saved_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                "image": res["image"],
                                "STORE": store_val.strip() or None,
                                "DATE": date_val.strip() or None,
                                "TOTAL": int(total_val) if total_val > 0 else None,
                                "TOTAL_RAW": res["TOTAL_RAW"],
                                "angle": res["angle"]
                            })
                            st.session_state["save_status"] = "saved"
                            st.rerun()
                    with btn_c2:
                        if st.button(" Không lưu", use_container_width=True):
                            st.session_state["save_status"] = "dismissed"
                            st.rerun()

            elif st.session_state["save_status"] == "saved":
                st.success(" Đã lưu thông tin hóa đơn vào tab **Lịch Sử Hóa Đơn**!")
            elif st.session_state["save_status"] == "dismissed":
                st.caption("Đã bỏ qua lưu hóa đơn này.")

            st.divider()

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("**Ảnh sau khi tự động xoay hướng**")
                st.image(output["oriented_image"], use_container_width=True)
            with img_col2:
                st.markdown("**Ảnh gán nhãn Bounding Box (LayoutLMv3)**")
                st.image(output["annotated_image"], use_container_width=True)

            with st.expander(" Xem bảng chi tiết các từ (OCR Tokens & Nhãn dự đoán)"):
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

    with tab_batch:
        st.write(f"Tìm thấy **{len(existing_images)}** ảnh trong thư mục `{INPUT_DIR}`.")

        if st.button(" Chạy toàn bộ ảnh trong demo_input"):
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
                    label=" Tải xuống kết quả CSV",
                    data=csv_bytes,
                    file_name="extracted_results_demo.csv",
                    mime="text/csv"
                )

    with tab_history:
        st.subheader(" Lịch Sử Các Hóa Đơn Đã Lưu")

        if not history_data:
            st.info("Chưa có hóa đơn nào được lưu. Hãy trích xuất hóa đơn ở Tab 1 và nhấn **'Có, lưu hóa đơn'**.")
        else:
            df_hist = pd.DataFrame(history_data)
            df_hist["TOTAL_NUM"] = pd.to_numeric(df_hist["TOTAL"], errors="coerce").fillna(0).astype(int)

            sum_c1, sum_c2 = st.columns(2)
            sum_c1.metric(" Tổng số hóa đơn đã lưu", f"{len(df_hist)} hóa đơn")
            sum_c2.metric(" Tổng chi tiêu cộng dồn", f"{df_hist['TOTAL_NUM'].sum():,} đ")

            st.caption(" Tích vào ô **'Chọn xóa'** ở đầu các dòng bạn muốn bỏ, sau đó nhấn nút **'Xóa mục đã chọn'** bên dưới.")

            df_display = df_hist.drop(columns=["TOTAL_NUM"]).rename(columns={
                "saved_at": "Thời gian lưu",
                "image": "Tên file ảnh",
                "STORE": "Cửa hàng (STORE)",
                "DATE": "Ngày mua (DATE)",
                "TOTAL": "Tổng tiền (VNĐ)",
                "TOTAL_RAW": "Tổng tiền gốc (RAW)",
                "angle": "Góc xoay (°)"
            })
            df_display.insert(0, "Chọn xóa", False)

            edited_df = st.data_editor(
                df_display,
                hide_index=True,
                use_container_width=True,
                disabled=[col for col in df_display.columns if col != "Chọn xóa"],
                key="history_editor"
            )

            selected_indices = edited_df.index[edited_df["Chọn xóa"]].tolist()

            act_c1, act_c2, act_c3, _ = st.columns([1.2, 1.2, 1.2, 2.4])
            with act_c1:
                if st.button(
                    f" Xóa mục đã chọn ({len(selected_indices)})",
                    disabled=(len(selected_indices) == 0),
                    use_container_width=True
                ):
                    remaining_records = [
                        rec for idx, rec in enumerate(history_data)
                        if idx not in selected_indices
                    ]
                    rewrite_history(remaining_records)
                    st.rerun()

            with act_c2:
                hist_csv_bytes = df_hist.drop(columns=["TOTAL_NUM"]).to_csv(
                    index=False, encoding="utf-8-sig"
                ).encode("utf-8-sig")
                st.download_button(
                    label=" Xuất lịch sử CSV",
                    data=hist_csv_bytes,
                    file_name="saved_receipts_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with act_c3:
                if st.button(" Xóa toàn bộ lịch sử", use_container_width=True):
                    clear_history()
                    st.rerun()


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        main()