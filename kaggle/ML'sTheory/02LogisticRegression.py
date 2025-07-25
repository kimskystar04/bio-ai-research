import torch
import torch.nn as nn
import torch.optim as optim

# 데이터: 논리 AND 연산 (0,1 이진 분류)
x_data = torch.tensor([[0.], [1.], [2.], [3.]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])

# 모델 정의
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # 시그모이드 통과

# 모델, 손실함수, 옵티마이저 설정
model = LogisticRegression()
criterion = nn.BCELoss()  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습 루프
for epoch in range(1000):
    output = model(x_data)
    loss = criterion(output, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:03}: Loss = {loss.item():.4f}")

# 테스트
test_input = torch.tensor([[1.5]])
with torch.no_grad():
    pred = model(test_input)
    print(f"\n입력 1.5에 대한 예측 확률: {pred.item():.4f}")
    print("클래스 예측:", int(pred.item() > 0.5))
