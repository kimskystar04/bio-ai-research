# PCA, SVD 실습
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 생성 (2D Gaussian)
np.random.seed(42)
mean = [0, 0]
cov = [[3, 1], [1, 2]]  # 상관관계 있는 2차원 데이터
X = np.random.multivariate_normal(mean, cov, 200)
print(X)
# 2. 평균 중심화
X_centered = X - np.mean(X, axis=0)

# 3. SVD 수행
U, S, Vt = np.linalg.svd(X_centered)

# 4. 주성분 추출 (Vt[0]이 1st PC)
first_pc = Vt[0]  # 첫 번째 주성분 방향

# 5. 투영 (2D → 1D)
X_pca_1D = X_centered @ first_pc.reshape(-1, 1)  # 주성분 축으로 투영
X_projected = X_pca_1D @ first_pc.reshape(1, -1)  # 원래 공간으로 복원

# 6. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3, label='Original Data')
plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.8, label='Projected (1D)')
plt.quiver(0, 0, first_pc[0], first_pc[1], color='r', scale=3, label='First PC')
plt.axis('equal')
plt.legend()
plt.title('PCA via SVD (2D → 1D)')
plt.grid(True)
plt.show()
