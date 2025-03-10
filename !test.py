import torch
import torch.nn as nn

# 모델 클래스 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 모델 생성 및 저장
model = NeuralNetwork()
torch.save(model, 'model.pth')  # 모델 전체 저장

# 모델 로드 (클래스 정의가 필요)
model = torch.load('model.pth')
