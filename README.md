# PyTorch  

## 2025-03-06: Tensor  ##  
1-1. Tensor의 속성에는 shape, dtype, device가 있다.  
1-2. dtype은 데이터 타입을, device는 Tensor가 저장되는 위치(cpu,gpu)가 기록된다.  
1-3. cpu에 저장되는 경우, Tensor와 Numpy의 배열은 동일한 메모리를 공유한다.  
1-4. torch.cuda.is_available() 을 통해 gpu가 사용 가능한지 확인할 수 있다.  
1-5. tensor.to("cuda") 를 통해 gpu로 이동할 수 있다.  

2-1. torch.~_like(tensor2) 를 통해 tensor2의 속성(shape, dtype)을 다른 텐서에 적용할 수 있다.  

3-1. tensor.sum() 을 통해 각 요소를 더할 수 있다.  
3-2. tensor.sum()의 dtype은 torch.Tensor 이다.  
3-3. (tensor.sum()).item() 을 통해 float로 형변환 가능하다.  

4-1. Numpy는 Tensor로, Tensor는 Numpy로 변환 가능하다.  

----

## 2025-03-06: Dataset & DataLoader  ##  
1-1. Dataset은 학습 데이터와 테스트 데이터를 저장하는 장소이다.  
1-2. Dataset은 학습할 때, 특징을 batch에 전달하고 정답을 출력한다.  

2-1. Dataset의 번거러운 학습 방식을 추상화 시킨 것을 DataLoader라고 한다.  
2-2. DataLoader는 batch_size를 통해 한번에 학습하는 데이터 수를 지정할 수 있다.  
2-3. batch_size가 크면 빈번한 업데이트가 발생해 정확도가 높아진다.  
2-4. 다만, 학습 속도가 느려지고 GPU의 효율성이 떨어진다.  

3-1. Torch의 Dataset에는 root(저장소), train(학습? 테스트?), download(root에 없을시 다운?), transform(형식) 지정  
3-1-1. '?' 는 True or False 의 값을 지정할 수 있다.  
3-2. 사용자 정의 파일에서 Dataset을 불러올 땐 \_\_init\_\_, \_\_len\_\_, \_\_getitem\_\_ 을 반드시 구현해야 한다.  
3-3. __getitem__은 주어진 index 인자를 통해 해당 index의 Label을 출력할 수 있다.  

----

## 2025-03-07: Transform ##  
1-1. 일반적인 데이터는 기계를 학습시키에 불순한 상태이다.  
1-2. transform을 통해 데이터를 가공할 수 있다.  

2-1. Feature(특징)은 Dataset에서 데이터를 불러올 때 transform으로 지정할 수 있다.  
2-2. 이를 통해 Feature은 Normalization 된다.  

3-1. Label(정답)은 Dataset에서 데이터를 불러올 때 target_transformd으로 지정할 수 있다.  
3-2. 이를 통해 Label은 One-Hot Encoding 형식이 된다.  
3-3. One-Hot Encoding == 0과 1로 구성 -> 정답인 요소만 1로 표현한다.  

----

## 2025-03-07: Neural Network ##  
1-1. PyTorch의 모든 모듈은 nn.Module의 기능을 상속받는다.  

2-1. nn.Sequential == 기능을 순차적으로 수행할 수 있다.  
2-2. nn.Linear == 은닉층을 생성하고, 출력 데이터 수를 변경할 수 있다.  
2-3. nn.ReLU (비선형 함수) == 복잡한 데이터를 학습할 수 있다.  
2-4. nn.Flatten() == 평탄화 작업 == nn.Linear에 데이터 전달한다.  

3-1. Softmax 함수를 통해 주어진 값을 확률분포 형태로 변형한다.  
3-2. 해당 확률에서 가장 큰 값을 반환한다.  

----

## 2025-03-08: Autograd ##  
1-1. 신경망은 역전파 알고리즘을 주로 사용한다.  
1-2. 역전파 AL == 오차값을 계산하고, 역방향으로 전파해 가중치 값을 조절한다.  
1-3. 역전파 AL의 가중치(w)는 Gradient에 의해 결정된다.  
1-4. PyTorch의 backward()를 통해 역전파를 수행하고, 손실함수의 기울기를 계산한다.  

2-1. 손실함수의 기울기를 구하는 이유 == 파라미터 최적화  
2-2. 계산된 기울기는 각 파라미터의 'grad' 속성에 저장된다.  
2-3. 성능 상의 이유로 backward()는 동일 그래프 내에서 1회 사용 가능하다.  
2-4. 여러번 수행해야 한다면, retain_graph=True 를 지정해야 한다.  

3-1. requires_grad()=? 는 Tensor의 기울기 계산 여부를 지정할 수 있다.  
3-2. 주로 학습 가능한 파라미터에 적용한다.  
3-3. Tensor의 연산 기록을 추적하고, 변화도 계산을 지원한다.  
3-4. torch.no_grad() 블럭으로 감싸면 추적을 중단할 수 있다.  

4-1. 추적을 중단하는 이유 ?  
4-2. 매개변수를 고정해야 하는 경우  
4-3. 순전파만 계산할 때, 속도 향상을 위해  

----

## 2025-03-08: Optimzer, Loss_Fn ##  
1-1. '하이퍼 파라미터' 는 모델의 학습에 영향을 줄 수 있다.  
1-2. 구성요소로는 Epochs, Learning Rate, Batch_Size 가 있다.  
1-3. Epoch의 각 단계는 학습 단계와 검증 (성능 개션 확인 여부) 단계로 구성되어 있다.  

2-1. 손실함수는 실제값과 예측값 간의 차이를 계산한 것 이다.  
2-2. 손실함수의 종류에는 대표적으로 Mean Squared Error, CrossEntropy가 있다.  

3-1. 옵티마이저는 오차값을 줄이기 위해 매개변수 값을 조정한다.  
3-2. 옵티마이저의 종류에는 확률적 경사 하강법(SGD), ADAM이 있다.  

4-1. 학습 또는 평가 할 때 model.train() , model.eval() 과 같이 명시해야 한다.  
4-2. 이는, Batch Normalization과 Dropout 레이어를 활성화 및 비활성화 할 수 있다.  

----

## 2025-03-09: AI Weekly 요약 ##  
1-1. 기사 실기 대비하느라 PyTorch 학습 연기 \
1-2. 아래는 아침에 읽었던 AI Weekly News Letter \
1-3. [링크] https://www.nytimes.com/2025/02/27/learning/what-students-are-saying-about-ai-and-the-future-of-work.html

2-1. '외국 학생들의 AI에 대한 평가 - 부정' \
2-1-1. 반복가능한 업무는 tech 관련 기술직 이라도 대체될 수 있다. \
2-1-2. 자동화에 취약하지 않은 일자리를 탐색해야 한다. \

2-2. '외국 학생들의 AI에 대한 평가 - 긍정' \
2-2-1. 감정에 따른 분석, 학습은 인간보다 떨어질 것 이다. \
2-2-2. AI에 의해 사라지는 직업보단, 새로 탄생되는 직업이 더 많을 것 이다. \
2-2-3. 과거 산업혁명 또는 컴퓨터가 급부상했을 때와 같이 현재 상황에 적응해야 한다. \
2-2-4. AI에 창의적, 감정적 업무 대신 육체적 노동을 부여하다. \
2-2-4-1. 그와 관련된 업무를 맡은 사람들은 생계를 위협받을 수 있다. \
2-2-4-2. 이 역시 적응의 문제인가 ? 

----

## 2025-03-10: Model Save, Load ##
1-1. PyTorch 모델은 학습된 상태를 'state_dict' 라는 속성에 임시저장 한다. \
1-2. 이는 메모리에 저장된 상태로, 프로세스를 종료하면 기록이 날라간다. \
1-3. 이를 영구 저장하기 위해 사용하는 메소드가 torch.save() 이다.

2-1. torch.save(모델명, '파일명') 을 통해 학습 상태를 파일로 출력할 수 있다. \
2-2. torch.load('파일명') 을 통해 저장된 학습 상태를 불러올 수 있다. \
2-3. 모델 전체를 저장 및 호출할 수 있지만, 호환성의 문제로 권장하지 않는다.

