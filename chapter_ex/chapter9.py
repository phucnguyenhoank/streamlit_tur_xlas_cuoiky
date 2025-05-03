import cv2
import numpy as np

L = 256

def Erosion(imgin, ksize=45, shape=cv2.MORPH_RECT):
    w = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.erode(imgin, w)

def Dilation(imgin, ksize=3, shape=cv2.MORPH_RECT):
    w = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.dilate(imgin, w)

def OpeningClosing(imgin, ksize=3, shape=cv2.MORPH_RECT):
    w = cv2.getStructuringElement(shape, (ksize, ksize))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_OPEN, w)
    return cv2.morphologyEx(temp, cv2.MORPH_CLOSE, w)

def Boundary(imgin, ksize=3, shape=cv2.MORPH_RECT):
    w = cv2.getStructuringElement(shape, (ksize, ksize))
    temp = cv2.erode(imgin, w)
    return imgin - temp

def HoleFill(imgin, seed_point=(105, 297)):
    imgout = imgin.copy()
    M, N = imgout.shape
    mask = np.zeros((M + 2, N + 2), np.uint8)
    cv2.floodFill(imgout, mask, seed_point, L - 1)
    return imgout

def MyConnectedComponent(imgin, thresh=200, blur_ksize=7, base_color=150):
    ret, temp = cv2.threshold(imgin, thresh, L - 1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, blur_ksize)
    M, N = temp.shape
    dem = 0
    color = base_color
    for x in range(M):
        for y in range(N):
            if temp[x, y] == L - 1:
                mask = np.zeros((M + 2, N + 2), np.uint8)
                cv2.floodFill(temp, mask, (y, x), color)
                dem += 1
                color += 1
    print(f'Co {dem} thanh phan lien thong')
    a = np.zeros(L, np.int64)
    for x in range(M):
        for y in range(N):
            r = temp[x, y]
            if r > 0:
                a[r] += 1
    for r, count in enumerate(a):
        if count > 0:
            print(f'{r:4d}   {count:5d}')
    return temp.astype(np.uint8)

def ConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d thanh phan lien thong' % (dem-1) 
    print(text)

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(1, dem):
        print('%4d %10d' % (r, a[r]))
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label

def CountRice(imgin, ksize=81, thresh=None, min_area=100, blur_ksize=3, add_text=True):
    # Tạo phần tử cấu trúc cho phép toán hình thái học
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    # Thực hiện phép biến đổi Top-Hat
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    
    # Phân ngưỡng
    if thresh is None:
        ret, temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, temp = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)
    
    # Làm mịn ảnh bằng median blur
    temp = cv2.medianBlur(temp, blur_ksize)
    
    # Gán nhãn các thành phần liên thông
    dem, label = cv2.connectedComponents(temp)
    
    # Tính diện tích của từng thành phần
    a = np.zeros(dem, np.int64)
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            r = label[x, y]
            a[r] += 1
    
    # Lọc các thành phần nhỏ
    xoa = [r for r in range(1, dem) if a[r] < min_area]
    
    # Tạo mask cho các thành phần lớn
    mask = np.zeros_like(label, dtype=np.uint8)
    for r in range(1, dem):
        if r not in xoa:
            mask[label == r] = 255
    
    # Đếm lại số thành phần sau khi lọc
    dem, filtered_label = cv2.connectedComponents(mask)
    print(f'Có {dem - 1} hạt gạo')
    
    # Vẽ đường viền trên ảnh gốc
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR) if len(imgin.shape) == 2 else imgin.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    
    # Thêm văn bản nếu cần
    if add_text:
        text = f'Co {dem - 1} hat gao'
        cv2.putText(output, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return output
