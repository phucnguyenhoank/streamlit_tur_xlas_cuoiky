import cv2
import numpy as np
import matplotlib.pyplot as plt
from chapter_ex.chapter9 import CountRice

L = 256  # Định nghĩa mức xám tối đa


# === Chạy test ===
if __name__ == "__main__":
    # Đọc ảnh xám
    img = cv2.imread("./images/rice.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Không tìm thấy ảnh 'rice.png'")
        exit()

    result = CountRice(img)

    # Hiển thị ảnh qua các giai đoạn
    titles = ['Gốc', 'Top-hat', 'Threshold', 'Sau làm mượt', 'Kết quả']
    # images = [img, top_hat, thresh_img, bin_img, result]

    plt.figure(figsize=(15, 8))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(result, cmap='gray')
        # plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
