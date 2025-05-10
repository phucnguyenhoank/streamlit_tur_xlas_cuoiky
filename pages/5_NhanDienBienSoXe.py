import cv2
import av
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import supervision as sv
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load models
plate_model = YOLO("ndbs_models/best_plate_model.pt")
char_model = YOLO("ndbs_models/best_char_model.pt")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

st.title("📸 Nhận diện biển số")
mode = st.radio("Chọn chế độ", ["📷 Camera realtime", "🖼️ Upload ảnh"])

def recognize_plate(image):
    plate_results = plate_model.predict(image, conf=0.4)
    plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy()

    for box in plate_boxes:
        x1, y1, x2, y2 = map(int, box)
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        plate_img_resized = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        plate_img_resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        pil_image = Image.fromarray(cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB))
        char_result = char_model.predict(pil_image, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(char_result)

        detected_string = ""
        if hasattr(detections, 'data') and 'class_name' in detections.data:
            char_boxes = detections.xyxy
            labels = detections.data['class_name']
            items = list(zip(char_boxes[:, 0], labels))
            items.sort(key=lambda x: x[0])
            detected_string = ''.join(label for _, label in items)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, detected_string, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    return image

if mode == "🖼️ Upload ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        result_image = recognize_plate(image_bgr)
        st.image(result_image, caption="Kết quả", channels="BGR")

elif mode == "📷 Camera realtime":
    class RealtimePlateDetector(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result = recognize_plate(image)
            return result

    webrtc_streamer(
        key="plate-detect",
        video_transformer_factory=RealtimePlateDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
