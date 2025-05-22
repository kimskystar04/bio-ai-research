# 이미지 불러오기 및 정규화
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

#  샘플 이미지 생성 (실습용)
os.makedirs("assets", exist_ok=True)
fake_img = np.random.rand(64, 64) * 255  # 0~255 이미지
fake_img = fake_img.astype(np.uint8)
Image.fromarray(fake_img, mode='L').save("assets/sample.png")

#  1. 이미지 불러오기
img = Image.open("assets/sample.png")
print("원본 모드:", img.mode)  # 'L' or 'RGB'
print("크기:", img.size)

#  2. 넘파이 배열로 변환
arr = np.array(img)
print("배열 shape:", arr.shape)

#  3. 정규화 (0~255 → 0~1)
norm = arr / 255.0

#  4. 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(arr, cmap='Pastel1')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(norm, cmap='Pastel1')
plt.title("Notmalized Image")

plt.tight_layout()
plt.show()
