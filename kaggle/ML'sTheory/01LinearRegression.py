import torch
import torch.nn as nn
import torch.optim as optim

# 데이터: y = 2x + 1 (예시)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 모델 클래스 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성
model = LinearRegressionModel()

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(200):
    pred = model(x_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"[{epoch:03}] Loss: {loss.item():.4f}")

# 학습 결과 확인
x_test = torch.tensor([[5.0]])
print("예측 결과:", model(x_test).item())  # 거의 11에 가까운 값
