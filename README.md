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
1-1. Dataset은 학습 데이터와 테스트 데이터를 저장하는 곳  
1-2. Dataset은 학습할 때, 특징을 batch에 전달하고 정답을 출력  

2-1. Dataset의 번거러운 학습 방식을 추상화 시킨 것을 DataLoader라고 함  
2-2. DataLoader는 batch_size를 통해 한번에 학습하는 데이터 수를 지정 가능  
2-3. batch_size가 크면 빈번한 업데이트가 발생해 정확도가 높아짐  
2-4. 다만, 학습 속도가 느려지고 GPU의 효율성이 떨어짐  

3-1. Torch의 Dataset을 사용할 땐 root(저장소), train(학습? 테스트?), download(root에 없을시 다운?), transform(형식) 지정  
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
3-3. One-Hot Encoding == 0과 1로 구성 -> 정답인 요소만 1로 수정  

----

## 2025-03-07: Neural Network ##  
1-1. PyTorch의 모든 모듈은 nn.Module의 기능을 상속받는다.  

2-1. nn.Sequential == 기능을 순차적으로 수행할 수 있다.  
2-2. nn.Linear == 은닉층을 생성하고, 출력 데이터 수를 변경할 수 있다.  
2-3. nn.ReLU (비선형 함수) == 복잡한 데이터를 학습할 수 있다.  
2-4. nn.Flatten() == 평탄화 작업 == nn.Linear에 데이터 전달한다.  

3-1. Softmax 함수를 통해 주어진 값을 확률분포 형태로 변형한다.  
3-2. 해당 확률에서 가장 큰 값을 반환한다.  
