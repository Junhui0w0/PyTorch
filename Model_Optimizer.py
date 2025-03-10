import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST( # 학습용 데이터셋 불러오기
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST( # 테스트용 데이터셋 불러오기
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Dataset은 Feature을 Batch에 전달하고, 정답Label을 출력함
# 이런 복잡한 과정을 추상화 == DataLoader
train_dataloader = DataLoader(training_data, batch_size=64)
test_datalader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module): #모든 기능은 nn.Module의 하위에 속함
    def __init__(self): # 생성자 (최초 1회 수행)
        super().__init__() # nn.Module 상속
        self.flatten = nn.Flatten() # 평탄화.... -> nn.Linear()에 데이터 전달하기 위함 (Fully Connected)
        self.linear_relu_stack = nn.Sequential( # 순차적으로 아래 코드 수행
            nn.Linear(28*28, 512), # 은닉층 추가 // 출력 데이터 수를 724(28*28) ->  512로 변경
            nn.ReLU(), # 비선형 함수 = 더 복잡한 데이터 학습
            nn.Linear(512 ,512), 
            nn.ReLU(), 
            nn.Linear(512, 10) 
        )

    def forward(self, x): #forward는 자동 수행
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()


# 하이퍼파라미터 == 모델 학습과 수렴율에 영향
# Epochs, batch_size, learning rate 정의
learning_rate = 1e-3 # 학습률
batch_size = 64 # 한번에 처리하는 데이터 수
epochs = 5 # 반복수


# 최적화 단계
# 단계의 각 반복 == 에폭Epoch
# 하나의 에폭 == 학습 단계, 검증(성능 개선 여부 확인) 단계 로 구성


# 손실 함수 == 획득한 결과와 실제 값 사이의 틀린 정도 (낮다 == 오차가 적다)
# 학습되지 않은 신경망 -> 오류 제공 확률 높음
# 종류: nn.MSELoss // nn.NLLLoss // nn.CrossEntropyLoss 등
loss_fn = nn.CrossEntropyLoss() # 손실 함수 초기화


# 옵티마이저 == 오류 감소를 위한 매개변수 조정
# 종류: SGD // ADAM // RMSProp 등
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 이후, optimizer.step() 를 통해 역전파에서 수집된 변화로 매개변수 조정



#=============================================================================#
def train_loop(dataloader, model, loss_fn, optimizer): # 학습
    size = len(dataloader.dataset)
    model.train() # 모델을 학습 모델로 설정 (batch normalization & dropout 레이어 활성화)

    for batch, (X,y) in enumerate(dataloader): #enumerate == index랑 value 반환??
        # pred와 loss 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward() #손실함수 역전파 수행 -> 손실함수 기울기 계산 == 매개변수 최적화
        optimizer.step() # 파라미터 업데이트
        optimizer.zero_grad() # 변화도 재설정 (중복 방지)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss :7f} [{current:>5d}/{size:>5d}]")

def test_loop(datalodaer, model, loss_fn): # 평가
    model.eval() # 모델을 평가 모델로 지정(batch normalization & Dropout 레이어 비활성화)
    size = len(datalodaer.dataset)
    num_batches = len(datalodaer)
    test_loss, correct = 0,0

    with torch.no_grad(): #Tensor 연산 추적 방지 -> 순전파만 계산 시 연산 속도 향상
        for X, y in datalodaer:
            pred = model(X)
            test_loss += loss_fn(pred ,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                #만약 예측된 값의 가장 큰 값이 정답Label(y)와 일치하면 Correct(정확도) 향상

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct) :>0.1f}%, AVG loss: {test_loss:>8f}\n")

loss_fn = nn.CrossEntropyLoss() # 분류에 주로 사용하는 손실함수
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 최적화 알고리즘(SGD)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_datalader, model, loss_fn)
    #model은 사전에 NeuralNetwork Class로 정의해서 동일한 인스턴스를 가지고 학습 및 평가 가능
print("DONE!")



#======================================================================#
# [정리]
# 1-1. 하이퍼 파라미터는 모델의 학습 정도에 영향을 미친다.
# 1-2. 하이퍼 파라미터의 종류에는 학습률, 에폭 수, 배치 사이즈 등이 있다.

# 2-1. 각 에폭 단계는 학습 단계와 성능 개선 확인 단계로 구성되어 있다.

# 3-1. 손실함수는 예측값과 실제값 간의 차이를 계산한 것 이다.
# 3-2. 손실함수의 종류에는 Mean Squared Error, CrossEntropy 등이 있다.

# 4-1. 옵티마이저는 오차를 줄이기 위해 매개변수를 조정하는 것 이다.
# 4-2. 옵티마이저의 종류에는 확률적 경사 하강법(SGD), ADAM 이 있다.

# 5-1. 학습과 평가 단계에선 model.train()과 model.eval() 를 명시해야 한다.
# 5-2. 이는, Batch Normalization과 Dropout 레이어를 (비)활성화 할 수 있다.
#======================================================================#
