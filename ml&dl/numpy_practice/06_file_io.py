# .txt, .csv 파일 입출력
import numpy as np
import os

#  1. 저장할 데이터 준비
data = np.array([[1.1, 2.2, 3.3],
                 [4.4, 5.5, 6.6]])

#  2. 텍스트(.txt), CSV 저장
np.savetxt("file_sample.txt", data)
np.savetxt("file_sample.csv", data, delimiter=",")

#  3. 텍스트 불러오기
loaded_txt = np.loadtxt("file_sample.txt")
loaded_csv = np.loadtxt("file_sample.csv", delimiter=",")

print("TXT 파일:\n", loaded_txt)
print("CSV 파일:\n", loaded_csv)

#  4. NPY (넘파이 전용 포맷) 저장/불러오기
np.save("file_array.npy", data)  # 저장
loaded_npy = np.load("file_array.npy")  # 불러오기
print("NPY 파일:\n", loaded_npy)

#  5. NPZ (여러 배열 압축 저장) 저장/불러오기
arr1 = np.array([1, 2, 3])
arr2 = np.array([[10, 20], [30, 40]])
np.savez("file_bundle.npz", one=arr1, two=arr2)

loaded_npz = np.load("file_bundle.npz")
print("NPZ 파일 (one):", loaded_npz['one'])
print("NPZ 파일 (two):\n", loaded_npz['two'])
