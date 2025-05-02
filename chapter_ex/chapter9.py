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

def ConnectedComponent(imgin, thresh=200, blur_ksize=7, base_color=150, add_text=True):
    ret, temp = cv2.threshold(imgin, thresh, L - 1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, blur_ksize)
    dem, label = cv2.connectedComponents(temp)
    print(f'Co {dem - 1} thanh phan lien thong')
    a = np.zeros(dem, np.int64)
    M, N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += base_color
    for r in range(1, dem):
        print(f'{r:4d} {a[r]:10d}')
    label = label.astype(np.uint8)
    if add_text:
        text = f'Co {dem - 1} thanh phan lien thong'
        cv2.putText(label, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return label

def CountRice(imgin, ksize=81, thresh=100, blur_ksize=3, base_color=150, add_text=True):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, thresh, L - 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, blur_ksize)
    dem, label = cv2.connectedComponents(temp)
    print(f'Co {dem - 1} hat gao')
    a = np.zeros(dem, np.int64)
    M, N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += base_color
    for r in range(dem):
        print(f'{r:4d} {a[r]:10d}')
    
    max_val = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max_val:
            max_val = a[r]
            rmax = r
    xoa = [r for r in range(1, dem) if a[r] < 0.5 * max_val]
    
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            if r > 0:
                r -= base_color
                if r in xoa:
                    label[x, y] = 0
    label = label.astype(np.uint8)
    if add_text:
        text = f'Co {dem - 1} hat gao'
        cv2.putText(label, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return label