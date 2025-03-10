# 일반적인 데이터는 학습에 바로 사용할 만큼 정제되어 있지 않음.
# 학습에 사용할 수 있도록 데이터를 정제하는 행위 == Transform

# Feature은 Torch에서 데이터를 불러올 때 사용하는 Transform 인자에 값을 대입해 Normalization을 수행
# Label은 target_transform으로 One-Hot Encoding 변환
# One-Hot Encoding == 0과 1로만 구성 -> 정답인 레이블만 1로 표시

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(), #Feature Normalization(정규화) // 픽셀크기를 0~1 사이로 조정
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        # Label을 One-Hot Encoding으로 변환
        # 0이 10개인 tensor에서, 정답인 레이블만 1로 변경
)


#------------------------------------------#
# 데이터를 정제하는 것 == Transform
# Feature은transform으로 Normalization
# Label은 target_transform으로 One-Hot Encoding 변환
#------------------------------------------#
