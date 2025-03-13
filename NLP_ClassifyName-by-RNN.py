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

class RNN(nn.Module): #RNN이란 모델은 아래 코드를 수행함.
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__() #RNN의 생성자 수행 (기존 RNN + 아래 기능 추가)

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size) #입력층 (출력 크기 input_size -> hidden_size)
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


# 네트워크 출력으로 가장 확률이 높은 카테고리의 이름과 번호 리턴
def catergoryFromOutput(output):
    top_n, top_i = output.topk(1) #텐서의 가장 큰 값과 주소
    category_i = top_i[0].item() #텐서 -> 정수 형변환
    return all_categories[category_i], category_i

print(f'catergoryFromOutput(output) : {catergoryFromOutput(output)}')

import random

def randomChocie(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChocie(all_categories)
    line = randomChocie(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10): #category = Country // line = 해당 국가에 맞는 이름
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print(f'category: {category} // line = {line}')


# 네트워크 학습
# RNN의 마지막 계층 == nn.LogSoftmax 이므로 손실함수는 nn.NLLLoss가 적합 (왜?)
# -> 수학적 특성과 안전성이 높기 때문.
# -> LogSoftmax는 입력 텐서에 Log를 수행한 후에 Softmax 함수 수행. (확률분포-LOG)
# -> NLLLoss는 로그-확률을 입력받아 음의 로그 우도를 계산함. 
# -> PyTorch의 CrossEntropy는 NLLLoss랑 LogSoftmax함수를 내부적으로 결합한 것
# 즉, CrossEntropy 하나만 사용하든, NLLLoss랑 LogSoftmax 둘이 같이 쓰든 해야 함

criterion = nn.NLLLoss()
lr = 0.005 # 학습률이 너무 높으면 발산할 수 있고, 너무 낮으면 학습 안할 수 있음 (적당한 값 탐색 필요)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward() #역전파 -> 손실함수의 기울기를 계산해서 grad 속성에 저장

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item()

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

cur_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s}s'

start = time.time()

for iter in range(1, n_iters+1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    cur_loss += loss

    if iter % print_every == 0:
        guess, guess_i = catergoryFromOutput(output)
        correct = '[O]' if guess == category else '[X] (%s)' % category
        print(f'{iter} {iter / n_iters * 100}% ({timeSince(start)}) {loss} {line} / {guess} {correct}')

        if iter % plot_every == 0:
            all_losses.append(cur_loss / plot_every)
            cur_loss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 사용자 입력으로 실행
def predict(input_line, n_predictions =3):
    print('/n> %s' % input_line)

    with torch.no_grad(): #추적 방지 -> 순전파 연산 속도 상승
        outpt = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
            #topk -> 가장 큰 값과 인덱스 추출
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) % s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Jackson')


#===========================================================#
# 1-1. 옵티마이저로 LogSoftmax 함수를 사용하면, 손실함수로는 NLLLoss 함수를 사용해야 함
# 1-2. LogSoftmax함수는 텐서를 Log 적용 후 Softmax함수로 확률분포 형태로 변환
# 1-3. NLLLoss 함수는 로그-확률 값의 음의 우도를 계산

# 2-1. NLLLoss와 LogSoftmax 함수를 결합한 것이 PyTorch의 CrossEntropy
# 2-2. topk() 함수는 최대 value와 해당 값의 index를 반환

# cf. 정보처리기사 실기 시험 당일까지 학습 연기
#===========================================================#