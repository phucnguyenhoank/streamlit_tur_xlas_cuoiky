import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import torchvision.transforms as transforms
import torch
import torch.nn as nn


# === Your existing config ===
original_classes = {1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 
4: 'Speed limit (60km/h)', 5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)', 
8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)', 10: 'No passing', 11: 'No passing veh over 3.5 tons', 
12: 'Right-of-way at intersection', 13: 'Priority road', 14: 'Yield', 15: 'Stop', 16: 'No vehicles', 
17: 'Veh > 3.5 tons prohibited', 18: 'No entry', 19: 'General caution', 20: 'Dangerous curve left', 
21: 'Dangerous curve right', 22: 'Double curve', 23: 'Bumpy road', 24: 'Slippery road', 
25: 'Road narrows on the right', 26: 'Road work', 27: 'Traffic signals', 28: 'Pedestrians', 
29: 'Children crossing', 30: 'Bicycles crossing', 31: 'Beware of ice/snow', 
32: 'Wild animals crossing', 33: 'End speed + passing limits', 34: 'Turn right ahead', 
35: 'Turn left ahead', 36: 'Ahead only', 37: 'Go straight or right', 38: 'Go straight or left', 
39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory', 42: 'End of no passing', 
43: 'End no passing veh > 3.5 tons'}

allowed_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42}
label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 15: 11, 16: 12, 17: 13, 
             32: 14, 33: 15, 34: 16, 35: 17, 36: 18, 37: 19, 38: 20, 39: 21, 40: 22, 41: 23, 42: 24}
inv_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)
IMG_SIZE = (30, 30)

# === Define your CNN model ===
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),  # 30x30 -> 26x26
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),  # 26x26 -> 22x22
            nn.ReLU(),
            nn.MaxPool2d(2),  # 22x22 -> 11x11
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3),  # 11x11 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),  # 9x9 -> 7x7
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3
            nn.Dropout(0.25)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# Load models
circle_model = YOLO(r"ndbb_models\best_circle_model.pt")
cnn_model = CNN()
cnn_model.load_state_dict(torch.load(r"ndbb_models\traffic_sign_classifier.pt", map_location=torch.device('cpu')))
cnn_model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# Traffic sign classification
def classify_sign(image):
    img = cv2.resize(image, (32, 32))  # Resize to input size
    tensor = transform(img).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = cnn_model(tensor)
        _, predicted = torch.max(output, 1)
        pred_label = predicted.item()
    label = original_classes[inv_label_map[pred_label]]
    return label

# Detection + classification
def detect_and_classify(image):
    results = circle_model.predict(image, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        try:
            label = classify_sign(cropped)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Class {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            print("Classification error:", e)

    return image

# Streamlit UI
st.title("🚦 Phân loại biển báo giao thông")
mode = st.radio("Chọn chế độ", ["🖼️ Upload ảnh", "📷 Camera realtime"])

if mode == "🖼️ Upload ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result_image = detect_and_classify(image_bgr)
        st.image(result_image, caption="Kết quả", channels="BGR")

elif mode == "📷 Camera realtime":
    class RealtimeDetector(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result = detect_and_classify(image)
            return result

    webrtc_streamer(
        key="realtime-sign-detect",
        video_transformer_factory=RealtimeDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
