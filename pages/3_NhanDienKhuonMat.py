import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import uuid
import os

# Load the pre-trained SVC model and dictionary of names
svc = joblib.load('model/svc.pkl')
mydict = ['HoangHuy', 'HuynhSon', 'NguyenHoangPhuc', 'TrungKy', 'VanLuan']

# Initialize face detector and recognizer
detector = cv.FaceDetectorYN.create(
    'model/face_detection_yunet_2023mar.onnx',
    "",
    (320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)
recognizer = cv.FaceRecognizerSF.create('model/face_recognition_sface_2021dec.onnx', "")

# Function to visualize faces and FPS
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, f'FPS: {fps:.2f}', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return input

# Video processor for real-time webcam detection
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tm = cv.TickMeter()
        self.frame_count = 0
        self.fps = 0

    def recv(self, frame):
        self.tm.start()
        img = frame.to_ndarray(format="bgr24")
        
        # Set input size for detector
        detector.setInputSize((img.shape[1], img.shape[0]))
        
        # Detect faces
        faces = detector.detect(img)
        
        # Process faces for recognition
        if faces[1] is not None:
            for face_box in faces[1]:
                x, y, w, h = face_box[:4].astype(np.int32)
                face_align = recognizer.alignCrop(img, face_box)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                cv.putText(img, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        self.tm.stop()
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.fps = self.tm.getFPS()
        self.tm.reset()
        
        # Visualize results
        img = visualize(img, faces, self.fps)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Face Recognition App")

# Webcam streaming
st.header("Real-time Webcam Detection")
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if st.button("Start/Stop Webcam"):
    st.session_state.streaming = not st.session_state.streaming

if st.session_state.streaming:
    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

# Video upload and processing
st.header("Upload Video for Face Detection")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    temp_file = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the video
    cap = cv.VideoCapture(temp_file)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((frame_width, frame_height))
    
    # Create a placeholder for video output
    stframe = st.empty()
    tm = cv.TickMeter()
    
    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            break
        
        tm.start()
        faces = detector.detect(frame)
        if faces[1] is not None:
            for face_box in faces[1]:
                x, y, w, h = face_box[:4].astype(np.int32)
                face_align = recognizer.alignCrop(frame, face_box)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                cv.putText(frame, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        tm.stop()
        frame = visualize(frame, faces, tm.getFPS())
        
        # Display frame in Streamlit
        stframe.image(frame, channels="BGR")
    
    cap.release()
    os.remove(temp_file)
    st.write("Video processing completed.")