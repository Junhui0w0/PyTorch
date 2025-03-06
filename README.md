# PyTorch  

**[2025-03-06]**  
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
