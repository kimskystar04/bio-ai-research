# 선형대수 (곱, 전치, 역행렬, 고윳값)
import numpy as np

# -------------------------------
# 행렬 정의
# -------------------------------
A = np.array([[2, 1],
              [1, 2]])

print("행렬 A:")
print(A)

# -------------------------------
# 행렬 전치 (Transpose)
# -------------------------------
print("\n전치행렬 A.T:")
print(A.T)

# -------------------------------
# 행렬 곱 (Dot Product)
# -------------------------------
B = np.array([[5, 6],
              [7, 8]])

print("\n행렬 B:")
print(B)

# 행렬곱 (A @ B) = np.dot(A, B)
print("\nA @ B:")
print(np.dot(A, B))  # 또는 A @ B

# -------------------------------
# 역행렬 (Inverse)
# -------------------------------
inv_A = np.linalg.inv(A)
print("\n역행렬 A⁻¹:")
print(inv_A)

# -------------------------------
# 행렬식 (Determinant)
# -------------------------------
det_A = np.linalg.det(A)
print("\n행렬식 det(A):", det_A)

# -------------------------------
# 고윳값 & 고유벡터
# -------------------------------
eigvals, eigvecs = np.linalg.eig(A)

print("\n고윳값 (Eigenvalues):")
print(eigvals)

print("\n고유벡터 (Eigenvectors):")
print(eigvecs)

# -------------------------------
# 대각화: A = P D P⁻¹
# -------------------------------
P = eigvecs
D = np.diag(eigvals)
P_inv = np.linalg.inv(P)

A_recon = P @ D @ P_inv

print("\n대각화 후 복원된 A (P D P⁻¹):")
print(A_recon)

# -------------------------------
# 확인용 오차 비교
# -------------------------------
print("\n원래 A와 복원된 A의 차이 (오차):")
print(A - A_recon)


