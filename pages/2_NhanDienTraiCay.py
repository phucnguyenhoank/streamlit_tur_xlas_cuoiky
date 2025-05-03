import streamlit as st
from PIL import Image
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Khởi tạo model
model = YOLO("weights/best.pt")  # Dùng "/" thay vì "\" cho tương thích hệ điều hành

# Tạo annotator
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

# Giao diện Streamlit
st.title("🍎 Ứng dụng nhận diện trái cây")

# Nút upload ảnh
uploaded_file = st.file_uploader("📤 Tải lên một ảnh", type=["jpg", "jpeg", "png"])

# Nếu người dùng đã upload
if uploaded_file is not None:
    # Mở ảnh từ file
    image = Image.open(uploaded_file).convert("RGB")

    # Hiển thị ảnh gốc
    st.subheader("🖼️ Ảnh gốc")
    st.image(image, use_container_width=True)

    # Chạy model dự đoán
    with st.spinner("🔍 Đang nhận diện..."):
        result = model.predict(image, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Annotate ảnh
        annotated_image = np.array(image.copy())  # Đổi sang mảng để supervision xử lý
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(annotated_image, detections=detections)

    # Hiển thị ảnh kết quả
    st.subheader("✅ Ảnh sau khi nhận diện")
    st.image(annotated_image, use_container_width=True)
