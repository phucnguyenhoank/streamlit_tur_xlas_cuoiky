import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from ultralytics import YOLO

# --- Khởi tạo model và thư viện ---
svc = joblib.load('model/svc.pkl')
labels = ['HoangHuy', 'HuynhSon', 'NguyenHoangPhuc', 'TrungKy', 'VanLuan']

# Face detector & recognizer (đường dẫn tương đối)
fd_model = 'model/face_detection_yunet_2023mar.onnx'
fr_model = 'model/face_recognition_sface_2021dec.onnx'
detector   = cv.FaceDetectorYN.create(fd_model, "",
                                      (320, 320), 0.9, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(fr_model, "")

# Hàm visualize
def visualize(frame, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[:4]
            score = face[-1]
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness)
            # keypoints
            for i,(kx,ky) in enumerate(zip(coords[4::2], coords[5::2])):
                cv.circle(frame, (kx,ky), 2, (255,0,0), thickness)
    cv.putText(frame, f'FPS: {fps:.2f}', (1,16),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

st.title("🪞 Face Detection & Recognition")

mode = st.radio("Chọn chế độ", ["Camera realtime", "Upload video"], index=0)

if mode == "Camera realtime":
    if st.button("▶️ Bắt đầu camera"):
        cap = cv.VideoCapture(0)
        frame_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frame_width, frame_height])
        tm = cv.TickMeter()

        st.info("Nhấn **q** trên cửa sổ OpenCV để dừng")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Không lấy được frame từ camera")
                break

            tm.start()
            faces = detector.detect(frame)
            tm.stop()

            # Nhận diện face + gán label
            if faces[1] is not None:
                for face_box in faces[1]:
                    x,y,w,h = face_box[:4].astype(int)
                    aligned = recognizer.alignCrop(frame, face_box)
                    feat    = recognizer.feature(aligned)
                    pred    = svc.predict(feat)[0]
                    name    = labels[pred]
                    cv.putText(frame, name, (x, y-10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # visualize bounding box + FPS
            out = visualize(frame, faces, tm.getFPS())

            cv.imshow("Camera - Nhấn q để thoát", out)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

else:  # Upload video
    uploaded = st.file_uploader("📹 Tải lên video", type=["mp4","avi","mov"])
    if uploaded is not None:
        # Lưu tạm video
        tfile = 'temp_video.' + uploaded.name.split('.')[-1]
        with open(tfile, 'wb') as f:
            f.write(uploaded.read())

        cap = cv.VideoCapture(tfile)
        frame_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frame_width, frame_height])
        tm = cv.TickMeter()

        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret: break

            tm.start()
            faces = detector.detect(frame)
            tm.stop()

            if faces[1] is not None:
                for face_box in faces[1]:
                    x,y,w,h = face_box[:4].astype(int)
                    aligned = recognizer.alignCrop(frame, face_box)
                    feat    = recognizer.feature(aligned)
                    pred    = svc.predict(feat)[0]
                    name    = labels[pred]
                    cv.putText(frame, name, (x, y-10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            out = visualize(frame, faces, tm.getFPS())
            # hiển thị trong Streamlit
            stframe.image(out, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Xử lý video xong!")
