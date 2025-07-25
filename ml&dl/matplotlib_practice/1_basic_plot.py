import matplotlib.pyplot as plt
import numpy as np
x = np.random.gamma(1,0.25,100000)


plt.hist(x,bins=100, color='blue', alpha=0.7,density=True, label="Data points")
plt.title("Scatter Plot Example")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()
plt.grid(True)
plt.show()

data = np.random.randn(1000)  # 평균 0, 표준편차 1 정규분포

plt.hist(data, bins=100, color='purple', edgecolor='black', alpha=0.8)
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()

categories = ['A', 'B', 'C', 'D']
values = [10, 24, 15, 20]

plt.bar(categories, values, color='orange', width=0.6)
plt.title("Bar Chart Example")
plt.xlabel("Category")
plt.ylabel("Value")
plt.grid(axis='y')
plt.show()