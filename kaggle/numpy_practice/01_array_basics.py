# 배열 생성, 형태, 인덱싱
import numpy as np

# 배열 생성
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])

# 기본 정보
print("a.shape:", a.shape)
print("b.ndim:", b.ndim) #span하는 차원이 아니라 구조적 배열의 공간을 말하는거 살짝 선대랑 다른 포인트!
print("b.dtype:", b.dtype)

# 배열 구조 변경
c = np.arange(12).reshape(3, 4)
print("Reshaped c:\n", c)

# 평탄화
print("Flatten:", c.flatten())
print("Ravel:", c.ravel())

# 인덱싱과 슬라이싱
print("c[1]:", c[1])         # 2번째 행
print("c[:, 2]:", c[:, 2])   # 3번째 열
print("c[1, 2]:", c[1, 2])   # (1,2) 원소
