import streamlit as st
import cv2
import numpy as np
import av
import random
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import hand_detector  # Đảm bảo bạn có hand_detector.py hoặc thư viện riêng

CHOICES = {0: "Paper", 1: "Rock", 2: "Scissors"}

def draw_results(frame, user_draw):
    com_draw = random.randint(0, 2)

    # Vẽ người chơi
    frame = cv2.putText(frame, 'You', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2, cv2.LINE_AA)
    try:
        # s_img = cv2.imread(os.path.join(f"{user_draw}.png"))
        s_img = cv2.imread(os.path.join(f"images\\keobuabao\\{user_draw}.png"))
        if s_img is not None:
            frame[100:100 + s_img.shape[0], 50:50 + s_img.shape[1]] = s_img
    except:
        pass

    # Vẽ máy
    frame = cv2.putText(frame, 'Bot', (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2, cv2.LINE_AA)
    try:
        # s_img = cv2.imread(os.path.join(f"{com_draw}.png"))
        s_img = cv2.imread(os.path.join(f"images\\keobuabao\\{com_draw}.png"))
        if s_img is not None:
            frame[100:100 + s_img.shape[0], 400:400 + s_img.shape[1]] = s_img
    except:
        pass

    # Tính kết quả
    if user_draw == com_draw:
        results = "DRAW"
    elif (user_draw == 0 and com_draw == 1) or \
         (user_draw == 1 and com_draw == 2) or \
         (user_draw == 2 and com_draw == 0):
        results = "YOU WIN"
    else:
        results = "YOU LOSE"

    h = frame.shape[0]
    frame = cv2.putText(frame, results, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 0, 255), 2, cv2.LINE_AA)

    return frame

import time

class RPSProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = hand_detector.handDetactor()
        self.show_result = False
        self.last_user_draw = -1
        self.result_frame = None
        self.result_timestamp = 0  # Lưu thời gian bắt đầu hiện kết quả
        self.result_duration = 2  # Thời gian giữ kết quả (giây)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Nếu đang hiển thị kết quả và chưa hết 2s, thì giữ nguyên ảnh kết quả
        if self.result_frame is not None and (time.time() - self.result_timestamp < self.result_duration):
            return av.VideoFrame.from_ndarray(self.result_frame, format="bgr24")

        img, hand_lms = self.detector.findHands(img)
        n_finger = self.detector.count_finger(hand_lms)

        user_draw = -1
        if n_finger == 5:
            user_draw = 0  # Paper
        elif n_finger == 0:
            user_draw = 1  # Rock
        elif n_finger == 2:
            user_draw = 2  # Scissors

        # Khi được yêu cầu hiển thị kết quả
        if self.show_result and user_draw != -1:
            self.last_user_draw = user_draw
            img = draw_results(img, user_draw)
            self.result_frame = img.copy()  # Lưu lại ảnh kết quả
            self.result_timestamp = time.time()
            self.show_result = False  # Reset trigger

        else:
            self.result_frame = None  # Xóa kết quả cũ nếu đã hết thời gian

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def toggle_result(self):
        self.show_result = True


# Giao diện Streamlit
st.title("Game Oẳn Tù Tì (Rock - Paper - Scissors)")
st.markdown("## 🕹️ Hướng dẫn chơi Oẳn Tù Tì bằng tay")
st.markdown("1. **Đưa tay vào khung hình** với số ngón tay:")
st.markdown("- ✊ **0 ngón**: Búa (Rock)")
st.markdown("- ✌️ **2 ngón**: Kéo (Scissors)")
st.markdown("- ✋ **5 ngón**: Bao (Paper)")
st.markdown("2. **Nhấn nút `Tiến xùm`** bên dưới để xem kết quả *trận đấu*.")
st.markdown("🎉 **Chúc bạn may mắn và chơi vui vẻ!**")

ctx = webrtc_streamer(
    key="rps-game",
    video_processor_factory=RPSProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)

# Nút chơi
if st.button("Tiến xùm!"):
    if ctx.video_processor:
        ctx.video_processor.toggle_result()
