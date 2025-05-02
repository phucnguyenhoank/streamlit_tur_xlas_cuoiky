import numpy as np
import cv2

L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Zero-padding and normalize
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin / (L - 1)
    fp *= (-1) ** np.fromfunction(lambda x, y: (x + y) % 2, (P, Q))
    
    # Compute DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Compute spectrum
    S = np.sqrt(F[:, :, 0]**2 + F[:, :, 1]**2)
    return np.clip(S, 0, L - 1).astype(np.uint8)

def FrequencyFilter(imgin, D0=60, n=2):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Zero-padding
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    fp *= (-1) ** np.fromfunction(lambda x, y: (x + y) % 2, (P, Q))
    
    # Compute DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Create Butterworth High Pass Filter
    u, v = np.indices((P, Q))
    Duv = np.sqrt((u - P // 2)**2 + (v - Q // 2)**2)
    H = np.where(Duv > 0, 1.0 / (1.0 + np.power(D0 / Duv, 2 * n)), 0)
    
    # Apply filter
    G = F.copy()
    G[:, :, 0] *= H
    G[:, :, 1] *= H
    
    # Inverse DFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    gp = g[:, :, 0]
    gp *= (-1) ** np.fromfunction(lambda x, y: (x + y) % 2, (P, Q))
    
    # Crop to original size
    imgout = gp[:M, :N]
    return np.clip(imgout, 0, L - 1).astype(np.uint8)

def CreateNotchRejectFilter(P, Q, uvs=[(44, 58), (40, 119), (86, 59), (82, 119)], D0=10, n=2):
    H = np.ones((P, Q), np.float32)
    u, v = np.indices((P, Q))
    
    for u_k, v_k in uvs:
        Duv = np.sqrt((u - u_k)**2 + (v - v_k)**2)
        H *= np.where(Duv > 0, 1.0 / (1.0 + np.power(D0 / Duv, 2 * n)), 0)
        Duv = np.sqrt((u - (P - u_k))**2 + (v - (Q - v_k))**2)
        H *= np.where(Duv > 0, 1.0 / (1.0 + np.power(D0 / Duv, 2 * n)), 0)
    
    return H

def DrawNotchRejectFilter(P=250, Q=180, uvs=[(44, 58), (40, 119), (86, 59), (82, 119)], D0=10, n=2):
    H = CreateNotchRejectFilter(P, Q, uvs, D0, n)
    return (H * (L - 1)).astype(np.uint8)

def RemoveMoire(imgin, uvs=[(44, 58), (40, 119), (86, 59), (82, 119)], D0=10, n=2):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Zero-padding
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    fp *= (-1) ** np.fromfunction(lambda x, y: (x + y) % 2, (P, Q))
    
    # Compute DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Apply Notch Reject Filter
    H = CreateNotchRejectFilter(P, Q, uvs, D0, n)
    G = F.copy()
    G[:, :, 0] *= H
    G[:, :, 1] *= H
    
    # Inverse DFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    gp = g[:, :, 0]
    gp *= (-1) ** np.fromfunction(lambda x, y: (x + y) % 2, (P, Q))
    
    # Crop to original size
    imgout = gp[:M, :N]
    return np.clip(imgout, 0, L - 1).astype(np.uint8)