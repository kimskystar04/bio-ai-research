# 평균, 분산, 표준편차, 공분산
import numpy as np

# -------------------------------
# 샘플 데이터 생성
# -------------------------------
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print("원본 데이터:\n", data)

# -------------------------------
# 평균, 분산, 표준편차
# -------------------------------
print("\n전체 평균:", np.mean(data))
print("전체 분산:", np.var(data))
print("전체 표준편차:", np.std(data))

# 축을 기준으로 연산
print("\n행별 평균 (axis=1):", np.mean(data, axis=1))
print("열별 분산 (axis=0):", np.var(data, axis=0))

# -------------------------------
# 누적합, 누적곱, 차분
# -------------------------------
print("\n누적합:", np.cumsum(data))
print("누적곱:", np.cumprod(data))
print("차분:", np.diff(data, axis=1))  # 옆열 간 차이

# -------------------------------
# 최대, 최소, 위치
# -------------------------------
print("\n최댓값:", np.max(data))
print("최댓값 위치 (argmax):", np.argmax(data))
print("최솟값:", np.min(data))
print("최솟값 위치 (argmin):", np.argmin(data))

# -------------------------------
# 조건부 마스킹 & np.where
# -------------------------------
mask = data > 5
print("\n5보다 큰 값만 출력:", data[mask])

# 조건에 따라 값 바꾸기: 5보다 작으면 0, 크면 그대로
print("조건부 치환 (np.where):\n", np.where(data < 5, 0, data))

# -------------------------------
# 정렬, 정렬된 인덱스
# -------------------------------
print("\n정렬된 데이터 (행 기준):\n", np.sort(data, axis=1))
print("정렬 인덱스 (열 기준):\n", np.argsort(data, axis=0))

# -------------------------------
# 퍼센타일, 중앙값
# -------------------------------
print("\n중앙값:", np.median(data))
print("90퍼센타일:", np.percentile(data, 90))


import numpy as np

# -------------------------------
# 1. 분산 vs 공분산 해석
# -------------------------------

# (2, 3) 행렬: 2개 변수(x, y), 3개 샘플
X = np.array([[1, 2, 3],    # 변수 x (3개 샘플)
              [4, 5, 6]])   # 변수 y (3개 샘플)

print("원본 행렬 X:\n", X)
print("shape:", X.shape)

# np.var: 전체 원소의 분산 (flatten 후 계산)
print("\n전체 분산 (np.var):", np.var(X))

# np.var with axis
print("열 단위 분산 (axis=0):", np.var(X, axis=0))
print("행 단위 분산 (axis=1):", np.var(X, axis=1))

# np.cov: 공분산행렬 (변수 간 관계)
cov_matrix = np.cov(X)
print("\n공분산 행렬 (np.cov):\n", cov_matrix)

# 해석:
# - np.var(X): 모든 원소를 단일한 1D 벡터처럼 보고 분산 계산
# - np.cov(X): 각 행 = 변수, 열 = 샘플 → 변수 간 관계 분석

# -------------------------------
# 2. 고차원 텐서 해석 예시
# -------------------------------

# (3, 2, 4) 텐서: 3개 샘플, 2개 변수, 4차원 특성
tensor = np.random.rand(3, 2, 4)
print("\n3차원 텐서 (3샘플, 2변수, 4특성): shape =", tensor.shape)

# 차원별 평균 보기
mean_dim0 = np.mean(tensor, axis=0)  # 3개 샘플 평균 → (2, 4)
mean_dim1 = np.mean(tensor, axis=1)  # 변수 기준 평균 → (3, 4)
mean_dim2 = np.mean(tensor, axis=2)  # 특성 기준 평균 → (3, 2)

print("\n샘플 평균 (axis=0): shape =", mean_dim0.shape)
print("변수 평균 (axis=1): shape =", mean_dim1.shape)
print("특성 평균 (axis=2): shape =", mean_dim2.shape)

# -------------------------------
# 3. 텐서 vs 행렬 vs 스칼라 구분
# -------------------------------

a = np.array(3.14)             # 0차원 → 스칼라
b = np.array([1, 2, 3])        # 1차원 → 벡터
c = np.array([[1, 2], [3, 4]]) # 2차원 → 행렬
d = np.random.rand(2, 3, 4)    # 3차원 → 텐서

print("\n스칼라:", a.shape)
print("벡터:", b.shape)
print("행렬:", c.shape)
print("텐서:", d.shape)

# -------------------------------
# summary
# -------------------------------

'''
넘파이에서 "차원"은 수학적 공간 구조가 아닌 데이터 축(axis)을 의미한다.

- 선형대수: 차원 = 벡터 공간 구조 (ex: ℝⁿ)
- 통계: 행 = 변수(x, y), 열 = 샘플 → 공분산 구조로 해석
- 넘파이: shape = 단순한 배열의 구조적 정보
- 3차원 이상이면 → 관습적으로 "텐서"라고 부름

정리:
- 분산은 "원소의 퍼짐"
- 공분산은 "변수 간 선형 상관->인과 관계가 아님!(통계적으로 매우 중요한 뽀인트)"
- 텐서는 "축마다 다른 해석이 붙는 n차원 데이터 용기(내지는 박스)"
'''
