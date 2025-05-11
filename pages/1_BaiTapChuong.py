import streamlit as st
from PIL import Image
import numpy as np
#include "chapter3.py" as chapter3
from chapter_ex import chapter3, chapter4, chapter9
from bt_chuong import chuong3, chuong4, chuong9

st.title("Bài tập chương")

# Upload image
uploaded_file = st.file_uploader("Chọn một ảnh màu...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh màu đã tải lên", use_container_width=True)
    
    # Convert to grayscale
    img_gray = image.convert('L')
    st.image(img_gray, caption="Phiên bản ảnh xám", use_container_width=True)
    img_gray_np = np.array(img_gray)  # Chuyển thành mảng numpy để xử lý

    # Tạo các tab cho từng chương
    chapter3_tab, chapter4_tab, chapter9_tab = st.tabs(["Chương 3", "Chương 4", "Chương 9"])

    # Tab Chương 3
    with chapter3_tab:
        st.header("Chương 3")
        option = st.selectbox(
            'Chọn thuật toán',
            ['Negative', 'Logarit', 'Power', 'PiecewiseLinear', 'Histogram', 'HistEqual', 
             'HistEqualColor', 'LocalHist', 'HistStat', 'BoxFilter', 'SmoothingGauss', 
             'Threshold', 'MedianFilter', 'Sharpen', 'UnSharpMasking', 'Gradient']
        )
        st.write('Bạn đã chọn:', option)
        
        if option == 'Negative':
            result = chapter3.Negative(img_gray_np)
        elif option == 'Logarit':
            result = chapter3.Logarit(img_gray_np)
        elif option == 'Power':
            gamma = st.slider("Gamma", 0.1, 10.0, 5.0)
            result = chapter3.Power(img_gray_np, gamma)
        elif option == 'PiecewiseLinear':
            result = chapter3.PiecewiseLinear(img_gray_np)
        elif option == 'Histogram':
            result = chapter3.Histogram(img_gray_np)
        elif option == 'HistEqual':
            result = chapter3.HistEqual(img_gray_np)
        elif option == 'HistEqualColor':
            img_color = image.convert('RGB')
            img_color_np = np.array(img_color)
            result = chapter3.HistEqualColor(img_color_np)
        elif option == 'LocalHist':
            result = chapter3.LocalHist(img_gray_np)
        elif option == 'HistStat':
            result = chapter3.HistStat(img_gray_np)
        elif option == 'BoxFilter':
            result = chapter3.BoxFilter(img_gray_np)
        elif option == 'SmoothingGauss':
            result = chapter3.SmoothingGauss(img_gray_np)
        elif option == 'Threshold':
            result = chapter3.Threshold(img_gray_np)
        elif option == 'MedianFilter':
            result = chapter3.MedianFilter(img_gray_np)
        elif option == 'Sharpen':
            result = chapter3.Sharpen(img_gray_np)
        elif option == 'UnSharpMasking':
            result = chapter3.UnSharpMasking(img_gray_np)
        elif option == 'Gradient':
            result = chapter3.Gradient(img_gray_np)
        
        else:
            result = img_gray_np  # Placeholder nếu thuật toán chưa được cài đặt
        st.image(result, caption="Kết quả Chương 3", use_container_width=True)
            

    # Tab Chương 4
    with chapter4_tab:
        st.header("Chương 4")
        option = st.selectbox(
            'Chọn thuật toán',
            ['Spectrum', 'FrequencyFilter', 'DrawNotchRejectFilter', 'RemoveMoire']
        )
        st.write('Bạn đã chọn:', option)
        
        if option == 'Spectrum':
            result = chapter4.Spectrum(img_gray_np)
        elif option == 'FrequencyFilter':
            result = chapter4.FrequencyFilter(img_gray_np)
        elif option == 'DrawNotchRejectFilter':
            result = chapter4.ApplyNotchFilter(img_gray_np)
        elif option == 'RemoveMoire':
            result = chapter4.RemoveMoire(img_gray_np)
        else:
            result = img_gray_np  # Placeholder nếu thuật toán chưa được cài đặt
        st.image(result, caption="Kết quả Chương 4", use_container_width=True)

    # Tab Chương 9
    with chapter9_tab:
        st.header("Chương 9")
        option = st.selectbox(
            'Chọn thuật toán',
            ['Erosion', 'Dilation', 'OpeningClosing', 'Boundary', 'HoleFill', 'ConnectedComponent', 'CountRice']
        )
        st.write('Bạn đã chọn:', option)
    
        if option == 'Erosion':
            ksize = st.slider("Kích thước kernel", 1, 50, 45)
            result = chapter9.Erosion(img_gray_np, ksize)
        elif option == 'Dilation':
            ksize = st.slider("Kích thước kernel", 1, 50, 3)
            result = chapter9.Dilation(img_gray_np, ksize)
        elif option == 'OpeningClosing':
            ksize = st.slider("Kích thước kernel", 1, 50, 5)
            result = chapter9.OpeningClosing(img_gray_np, ksize)
        elif option == 'Boundary':
            ksize = st.slider("Kích thước kernel", 1, 50, 5)
            result = chapter9.Boundary(img_gray_np, ksize)
        elif option == 'HoleFill':
            result = chapter9.HoleFill(img_gray_np)
        elif option == 'ConnectedComponent':
            result = chapter9.ConnectedComponent(img_gray_np)
        elif option == 'CountRice':
            result = chapter9.CountRice(img_gray_np)
        # Thêm các thuật toán khác nếu cần
        else:
            result = img_gray_np  # Placeholder nếu thuật toán chưa được cài đặt
        st.image(result, caption="Kết quả Chương 9", use_container_width=True)