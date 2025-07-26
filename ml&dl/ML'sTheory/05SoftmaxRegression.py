import torch
import torch.nn as nn
import torch.optim as optim

# 예시 데이터: 입력 2차원, 클래스 3개
x_train = torch.tensor([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [6.0, 9.0],
    [1.0, 0.6],
    [9.0, 11.0]
])
y_train = torch.tensor([0, 0, 1, 1, 2, 1])  # 클래스 라벨 (정수)

# 모델 정의
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# 모델, 손실 함수, 옵티마이저
model = SoftmaxClassifier(input_dim=2, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습 루프
for epoch in range(100):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"[{epoch}] Loss: {loss.item():.4f}")

# 테스트
test_input = torch.tensor([[3.0, 5.0]])
with torch.no_grad():
    logits = model(test_input)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)

    print("\n=== 테스트 결과 ===")
    print(f"입력값: {test_input.tolist()[0]}")
    print(f"예측 확률: {probs.numpy().round(4)}")
    print(f"예측 클래스: {pred_class.item()}")
