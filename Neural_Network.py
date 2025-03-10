# Pytorch의 모든 모듈은 nn.Module의 하위 클래스

import os
import torch
from torch import nn # 인공 신경망
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

print(f"사용중인 Device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # nn.Module의 기능 상속
        self.flatten = nn.Flatten() # Full Connected // nn.Linear에 데이터 전달하기 위함
        self.linear_relu_stack = nn.Sequential( # 아래 코드를 순차적 수행
            nn.Linear(28*28, 512), # 출력 크기를 28*28(724) 에서 512로 변경 (은닉층)
            nn.ReLU(), #비선형 함수 (더욱 더 복잡하 데이터 학습을 위함)
            nn.Linear(512 , 512), 
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x): # 직접 호출하지 않아도 자동 수행됨
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(f"\nNeural Network Model: \n {model}")

X = torch.rand(1, 28, 28, device=device) # batch_size는 1, 이미지 크기는 28*28
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # 가능성 예측 // tensor가 (a,b,c)일 때 dim=1 이므로 b에 softmax 적용
                                        #Softmax는 자연상수(e)를 이용해 확률분포 형태로 표현
print(f"\npred_probab: {pred_probab}")
    
y_pred = pred_probab.argmax(1) # 가장 큰 확률을 가진 값 반환
print(f"\n예측된 Class : {y_pred}")


input_img = torch.rand(3,28,28) # batch_size // 이미지 크기 28 * 28(pixel)
print(f"\n\ninput_img의 Size: {input_img.size()}")

flatten = nn.Flatten() 
flat_img = flatten(input_img)
print(f"2D 이미지: {flat_img}") # batch_size가 3 // 784(28*28) pixel 크기의 이미지가 연속된 배열로 표시

layer1 = nn.Linear(in_features=28*28, out_features=20) # 출력 크기를 724 -> 20
hidden1 = layer1(flat_img) # 은닉층
print(f"hidden1의 size: {hidden1.size()}") # batch_size는 3 // 뉴런 수는 20

# 비선형 함수 적용
print(f"\nReLU 함수 적용 전: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"\nReLU 함수 적용 후: {hidden1}\n\n")



#----------------------------------------------------------------------------------------
# [정리]
# 1-1) nn.Sequential 을 통해 순차적으로 기능을 수행할 수 있다.
# 1-2) nn.Linear는 은닉층을 생성한다.
# 1-3) nn.ReLU는 비선형 함수로 복잡한 데이터를 학습할 수 있다. (상황에 따라 다른 비선형 함수 사용해야 함)
# 1-4) Flatten을 통해 평탄화 작업을 수행해 Linear에 데이터를 전달할 수 있도록 해야 한다. 
    #(Fully Connected == 한 층의 모든 뉴런이 다음 뉴런 층과 연결된 상태)

# 2-1) nn.Module을 상속받은 class에서 forward 함수는 필수적이며, 자동 수행된다.

# 3-1) Softmax 함수는 주어진 값을 확률분포로 변경한다.
# 3-2) 변경된 확률 값에서 가장 큰 값을 에측된 값으로 출력된다.
#----------------------------------------------------------------------------------------