import torch
import torch.nn as nn
import torch.optim as optim

# 데이터
x_train = torch.tensor([[1.], [2.], [3.], [4.]])
y_train = torch.tensor([[3.], [5.], [7.], [9.]])


# 모델 정의
class RidgeRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = RidgeRegression()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

lmbda = 0.1  # 정규화 강도

for epoch in range(200):
    preds = model(x_train)
    mse_loss = criterion(preds, y_train)

    # L2 penalty 추가
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)

    loss = mse_loss + lmbda * l2_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"[{epoch}] Loss: {loss.item():.4f}")
x_test = torch.tensor([float(input())])
print("예측 결과:", model(x_test).item())
