# PyTorch  

<<<<<<< HEAD
## 2025-03-06: Tensor  
=======
**[2025-03-06]**  
>>>>>>> 65e2225e208821d64b9a25988b9240f097e3ded9
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
<<<<<<< HEAD

## 2025-03-06: Dataset & DataLoader  
1-1. Dataset은 학습 데이터와 테스트 데이터를 저장하는 곳  
1-2. Dataset은 학습할 때, 특징을 batch에 전달하고 정답을 출력  

2-1. Dataset의 번거러운 학습 방식을 추상화 시킨 것을 DataLoader라고 함  
2-2. DataLoader는 batch_size를 통해 한번에 학습하는 데이터 수를 지정 가능  
2-3. batch_size가 크면 빈번한 업데이트가 발생해 정확도가 높아짐  
2-4. 다만, 학습 속도가 느려지고 GPU의 효율성이 떨어짐  

3-1. Torch의 Dataset을 사용할 땐 root(저장소), train(학습? 테스트?), download(root에 없을시 다운?), transform(형식) 지정  
3-2. 사용자 정의 파일에서 Dataset을 불러올 땐 _\_init__, _\_len__, _\_getitem__ 을 반드시 구현해야 한다.  
3-3. __getitem__은 주어진 index 인자를 통해 해당 index의 Label을 출력할 수 있다.  
=======
>>>>>>> 65e2225e208821d64b9a25988b9240f097e3ded9
