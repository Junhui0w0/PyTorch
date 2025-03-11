# 캡스톤디자인2로 선정된 댓글(텍스트) 분석 프로젝트에 대비해 텍스트 파트 선택
# 데이터 전처리 과정에 대해 학습 할 예정 

#====[데이터 준비]====#
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)
print(findFiles('NLP_data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def unicodeToAscii(s): #unicode -> Ascii
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories=[]

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('NLP_data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


#====[이름 -> Tensor화]====#
# 하나의 문자를 표현하기 위해 One-Hot 벡터 사용
# One-Hot Vec == [0,0,0,1,0...,0] (정답만 1, 그외는 0)
    # -> 앞서 정답Label을 One-Hot Encoding 할 때 target_transform했는데 이와 관련있나 ?

import torch

# 문자 주소(index) 찾기
def letterToIndex(letter):
    return all_letters.find(letter)

# 검증 -> 한 문자를 <1xn_letters> 텐서로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line): #li = index // letter = value
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(f"\nletterToTensor('J') = {letterToTensor('J')} \n") #One-Hot Vector 확인
print(f"lineToTensor('Jones').size() = {lineToTensor('Jones').size()}\n\n")


#====[네트워크 생성]====#
# 본래는 nn.RNN이 구현되어 있으나, 학습의 목적으로 직접 RNN 사용할 예정
# RNN == 입력, 은닉상태, LogSoftmax(출력 뒤 동작) 3개의 선형 계층 구조

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__() #RNN의 생성자 수행

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size) #입력
        self.h2h = nn.Linear(hidden_size, hidden_size) #히든
        self.h2o = nn.Linear(hidden_size, output_size) #출력
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden): #자동수행
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output, '\n')



#==========================================================================#
# 1-1. 텍스트 데이터의 전처리 및 학습 과정에 대해 탐색

# 2-1. 텍스트 데이터를 학습시키기 위해 One-Hot Vector로 변형
# 2-2. One-Hot Vector == 정답은 1, 그외의 값들은 전부 0으로 구성

# 3-1. 네트워크 형성은 RNN을 직접 구현해 사용
# 3-2. RNN은 입력, 은닉, 출력 뒤 동작(LogSoftmax) 계층으로 구성
#==========================================================================#