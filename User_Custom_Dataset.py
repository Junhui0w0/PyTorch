# 파일에서 사용자 정의 데이터셋 제작
# 사용자 정의 Dataset 클래스는 !반드시! __init__, __len__, __getitem__ 을 구현해야 함.

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomeImageDataset(Dataset):
    # 생성자 최초 1회 수행
    def __init__(self, annotation_file, img_dir, transform=None, traget_transform=None):
        self.img_labels = pd.read_csv(annotation_file, names=['file_name', 'label']) 
            #인수로 주어진 annotationfile에서 이름이 file_name과 label 인 것을 읽어 img_labels에 저장
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = traget_transform

    # 길이 반환
    def __len__(self):
        return len(self.img_labels) 
    
    # 주어진 인덱스(idx)에 해당하는 샘플을 데이터셋에서 불러오고 반환
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path) # 이미지를 텐서로 변환
        label = self.img_labels.iloc[idx, 1] 

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label