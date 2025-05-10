import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

NUM_CLASSES = 4
IMG_SIZE = (30, 30)
allowed_labels = {14, 33, 34, 35}
label_map = {14: 0, 33: 1, 34: 2, 35: 3}

# -------------------------
# Load mô hình
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model(model_path="./ndbb_models/traffic_sign_model.pth"):
    model = TrafficSignNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -------------------------
# Danh sách lớp
classes = {0: 'Stop', 1: 'Turn right ahead', 2: 'Turn left ahead', 3: 'Ahead only'}

# -------------------------
# Hàm detect circle và crop
def detect_and_crop_circular_sign(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        margin = int(r * 1.2)
        x_min = max(x - margin, 0)
        x_max = min(x + margin, frame.shape[1])
        y_min = max(y - margin, 0)
        y_max = min(y + margin, frame.shape[0])
        cropped = frame[y_min:y_max, x_min:x_max]
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cropped_resized = cropped_pil.resize((30, 30))
        return np.array(cropped_resized), (x, y, r)
    return None, None

# -------------------------
# Hàm dự đoán từ ảnh
def predict_image(image_np):
    cropped_sign, circle_info = detect_and_crop_circular_sign(image_np)
    label_text = "No sign detected"
    if cropped_sign is not None:
        img_tensor = torch.tensor(cropped_sign.astype('float32') / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label_text = classes.get(predicted.item(), "Unknown")
        if circle_info:
            x, y, r = circle_info
            cv2.circle(image_np, (x, y), r, (0, 255, 0), 2)
    cv2.putText(image_np, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return image_np

# -------------------------
# Giao diện Streamlit
st.title("🚦 Nhận diện biển báo giao thông")

mode = st.radio("Chọn chế độ", ["🖼️ Upload ảnh", "📷 Camera realtime"])

if mode == "🖼️ Upload ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result = predict_image(image_bgr)
        st.image(result, channels="BGR", caption="Kết quả nhận diện")

elif mode == "📷 Camera realtime":
    class CameraProcessor(VideoTransformerBase):
        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result_frame = predict_image(image)
            return av.VideoFrame.from_ndarray(result_frame, format="bgr24")
        # def transform(self, frame):
        #     image = frame.to_ndarray(format="bgr24")
        #     return predict_image(image)

    webrtc_streamer(
        key="camera-detect",
        video_processor_factory=CameraProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

