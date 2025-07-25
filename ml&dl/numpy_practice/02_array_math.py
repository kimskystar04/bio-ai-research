# 기본 연산, 브로드캐스팅




import numpy as np

# 배열 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 기본 연산
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a ** 2 =", a ** 2)

# 스칼라 연산
print("a * 10 =", a * 10)

# 브로드캐스팅 예제
c = np.array([[1], [2], [3]])  # (3,1)
d = np.array([10, 20, 30])     # (3,)
print("Broadcasting c + d:\n", c + d)

# 집계 함수
print("sum:", np.sum(b))
print("mean:", np.mean(b))
print("max:", np.max(b))
print("argmax (최댓값 위치):", np.argmax(b))
print("cumsum:", np.cumsum(b))
print("prod:", np.prod(b))  # 모든 원소 곱

# -------------------------------
# 브로드캐스팅 시연: 열벡터 + 가로벡터
# -------------------------------

# 열벡터처럼 생긴 2D 배열 (3행 1열)
# shape: (3, 1)

# 1D 가로 벡터
# shape: (3,)

# 배열 형태 출력
print("c의 shape:", c.shape)  # (3, 1)
print("d의 shape:", d.shape)  # (3,)

# 브로드캐스팅 연산
# 넘파이는 내부적으로:
#   c: (3, 1)
#   d: (1, 3)로 확장함
# → 최종적으로 (3, 3) 배열로 연산 수행
result = c + d

# 결과 출력
print("\n브로드캐스팅 결과 (c + d):")
print(result)

# 해석 출력
print("\n해석:")
print("넘파이는 아래처럼 연산을 수행함:")
print("[[1],      +   [10 20 30]  →  [[11 21 31]")
print(" [2],                          [12 22 32]")
print(" [3]]                          [13 23 33]]")

# 개념 요약
print("\n요약 정리:")
print("- 넘파이 브로드캐스팅은 shape을 뒤에서부터 맞춘다")
print("- (3, 1) + (3,) → (3, 1) + (1, 3)로 확장 → 결과는 (3, 3)")
print("- 세로벡터 + 가로벡터 = 외적 느낌의 2D 행렬 생성")
print("- 이 방식은 딥러닝에서 bias 더할 때, 이미지 연산, 벡터화에 자주 사용됨")
