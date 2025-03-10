# Dataset에는 샘플과 정답을 저장
# DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 Iterable로 감싸는 것

# Iterable 이란?
    # __iter__() 메소드를 가진 객체
    # iter() 함수를 통해 이터레이터를 반환하는 객체
    # 파이썬의 List, Dictionary, Tuple, Set 등이 이에 해당
    # __next__() 메소드를 가지지 않지만, 이터레이터를 통해 순회 가능


# FashionMNIST 테스트
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Dataset 받아오기
training_data = datasets.FashionMNIST(
    root="data", #학습 및 테스트 데이터가 저장되는 경로
    train=True, #학습 or 테스트용 데이터셋 여부 지정 -> train=True ... 학습용 데이터셋
    download=True, #root에 데이터 없는 경우 다운
    transform=ToTensor() #Feature와 Label(정답)의 Transform 지정
)

test_data = datasets.FashionMNIST(
    root="data", 
    train=False, #train=False ... test용 데이터셋
    download=True,
    transform=ToTensor()
)


# Dataset 순회 및 시각화
labels_map = {
    0: "T-shirt",
    1: "Trouse",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"}


figure = plt.figure(figsize=(8,8)) #전체 Figure의 영역을 8*8(inch)로 설정
cols, rows = 3,3 # -> 전체 figure가 8*8 // 근데 3x3 행렬을 만듬 // 1x1 행렬의 변의 길이는 8/3 inch?

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
        #0 ~ training_data의 갯수 사이의 정수 1개를 뽑아 sample_idx에 저장
    img, label = training_data[sample_idx] #img는 사진 , label은 정답 (labels_map에서 img에 해당하는 idx)
    figure.add_subplot(rows, cols, i) 

    plt.title(labels_map[label])
    plt.axis("off") #축 라인, 눈금, 라벨 제거 -> 시각적 요소 제거
    plt.imshow(img.squeeze(), cmap="gray") #img.squeeze() -> Tensor의 불필요한 차원 제거
                                        #ex) 데이터 형태가 (1,H,W) -> (H,W)로 차원 제거

plt.show()


# DataLoader로 학습용 데이터 준비

# Dataset은 특징을 가져오고, 각 샘플에 정답을 지정하는 일을 한번에 수행
# 학습할 때, 샘플들을 minibatch에 전달 + 매 Epoch 마다 데이터를 다시 섞어 과적합 방지

# 이런 복잡한 과정을 추상화시킨 것 == DataLoader

from torch.utils.data import DataLoader

train_datalodader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True) 
    # shuffle=True...모든 배체 순회 뒤 데이터 섞임
    # batch_size ... 한번에 처리 할 데이터 수
    # batch_size가 크다 == 한번에 처리하는 데이터가 많다 == 빈번한 업데이트 발생 == 정확도 상승
    # 단, 한번에 처리할 데이터가 많아 속도가 느려질 수 있다 & GPU를 충분히 사용할 수 없음
    # 대부분 64부터 시작하며, 2배씩 높여 가장 좋은 성능이 나오는 값 확인 (64, 128, 256...)

train_features, train_labels = next(iter(train_datalodader))
print(f"Feature batch shape: {train_features.size()}\n")
    #return 값 -> [batch_size, 채널 수 (흑백이면 1), Height, Width]

print(f"Labels batch size: {train_labels.size()}\n")

img = train_features[0].squeeze() # 차원 축소
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}, type: {type(label)}")
print(f"Guessed Img: {labels_map[int(label)]}")


# -----------------------------------------------------------
# [정리]
# 1-1) Dataset은 학습데이터와 테스트 데이터를 저장하는 곳
# 1-2) Dataset은 학습할 때, 특징을 배치에 전달하고 정답을 출력함

# 2-1) DataLoader == Dataset의 불필요한 학습과정을 단순화 시킨 것
# 2-2) DataLoader는 Iterable 객체
# 2-3) DataLoader의 batch_size를 통해 한번에 학습할 수 있는 데이터 수 지정 가능
# 2-4) batch_size가 크면 빈번한 업데이트 발생 (= 정확도 상승)
# 2-5) 단, 속도 저하 및 GPU 성능 활용 X
# 2-6) shuffle을 통해 매 에폭 마다 데이터 섞임 여부 지정

# 3-1) torch의 Dataset 사용시 root (데이터 저장소), train(학습데이터? 테스트데이터?), download(root에 없을 경우 다운?), transform 지정
# 3-2) 사용자 지정 파일에서 Dataset을 정의할 경우 __init__, __len__, __getitem__을 반드시 구현해야 함
# 3-3) __getitem__은 인자로 주어진 idx에 해당하는 레이블 반환
# -----------------------------------------------------------