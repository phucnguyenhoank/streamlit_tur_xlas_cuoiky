import numpy as np
import cv2

L = 256

def Negative(imgin):
    return L - 1 - imgin

def Logarit(imgin):
    c = (L - 1) / np.log(L)
    imgout = c * np.log(1.0 + np.where(imgin == 0, 1, imgin))
    return imgout.astype(np.uint8)

def Power(imgin, gamma=5.0):
    c = np.power(L - 1, 1 - gamma)
    imgout = c * np.power(imgin, gamma)
    return imgout.astype(np.uint8)

def PiecewiseLinear(imgin):
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = max(rmin, 1)
    s1 = 0
    r2 = min(rmax, L - 2)
    s2 = L - 1
    imgout = np.zeros_like(imgin, dtype=np.float32)
    mask1 = imgin < r1
    mask2 = (imgin >= r1) & (imgin < r2)
    mask3 = imgin >= r2
    imgout[mask1] = (s1 / r1) * imgin[mask1]
    imgout[mask2] = (s2 - s1) / (r2 - r1) * (imgin[mask2] - r1) + s1
    imgout[mask3] = (L - 1 - s2) / (L - 1 - r2) * (imgin[mask3] - r2) + s2
    return imgout.astype(np.uint8)

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8) + 255
    h = np.histogram(imgin, bins=L, range=(0, L))[0]
    p = h / (M * N)
    scale = 2000
    for r in range(L):
        cv2.line(imgout, (r, M - 1), (r, M - 1 - int(scale * p[r])), (0, 0, 0))
    return imgout

def HistEqual(imgin):
    return cv2.equalizeHist(imgin)

def HistEqualColor(imgin):
    B, G, R = cv2.split(imgin)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    return cv2.merge((B, G, R))

def LocalHist(imgin, m=3, n=3):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    a, b = m // 2, n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            imgout[x, y] = cv2.equalizeHist(w)[a, b]
    return imgout

def HistStat(imgin, m=3, n=3):
    M, N = imgin.shape
    imgout = imgin.copy()
    mG, sigmaG = cv2.meanStdDev(imgin)
    a, b = m // 2, n // 2
    C, k0, k1, k2, k3 = 22.8, 0.0, 0.1, 0.0, 0.1
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            msxy, sigmasxy = cv2.meanStdDev(w)
            if (k0 * mG <= msxy <= k1 * mG) and (k2 * sigmaG <= sigmasxy <= k3 * sigmaG):
                imgout[x, y] = np.uint8(C * imgin[x, y])
    return imgout

def BoxFilter(imgin, m=11, n=11):
    w = np.ones((m, n)) / (m * n)
    return cv2.filter2D(imgin, cv2.CV_8UC1, w)

def SmoothingGauss(imgin, m=51, n=51, sigma=7.0):
    a, b = m // 2, n // 2
    w = np.fromfunction(lambda s, t: np.exp(-((s - a) ** 2 + (t - b) ** 2) / (2 * sigma ** 2)), (m, n))
    w /= w.sum()
    return cv2.filter2D(imgin, cv2.CV_8UC1, w)

def Threshold(imgin, ksize=15, thresh=64):
    temp = cv2.blur(imgin, (ksize, ksize))
    _, imgout = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)
    return imgout

def MedianFilter(imgin, m=5):
    return cv2.medianBlur(imgin, m)

def Sharpen(imgin):
    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = np.clip(imgin - temp, 0, L - 1)
    return imgout.astype(np.uint8)

def UnSharpMasking(imgin, k=10.0):
    blur = cv2.GaussianBlur(imgin, (3, 3), 1.0)
    mask = imgin.astype(np.float64) - blur
    imgout = np.clip(imgin + k * mask, 0, L - 1)
    return imgout.astype(np.uint8)

def Gradient(imgin):
    wx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = cv2.filter2D(imgin, cv2.CV_32FC1, wx)
    gy = cv2.filter2D(imgin, cv2.CV_32FC1, wy)
    imgout = np.clip(abs(gx) + abs(gy), 0, L - 1)
    return imgout.astype(np.uint8)