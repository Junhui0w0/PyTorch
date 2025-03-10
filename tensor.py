import torch
import numpy as np

#예제 데이터로부터 직접 Tensor 생성
data = [[1,2], [3,4]]
x_data = torch.tensor(data) #data의 자료형은 알아서 유추됨

# Tensor는 Numpy 배열로 생성 가능
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr) 

# 기본 속성 확인
tensor = torch.rand(3,4)
print(f"Tensor Shape: {tensor.shape}")
print(f"Tensor 데이터타입: {tensor.dtype}")
print(f"Tensor 저장 위치: {tensor.device}")

# 속성 변경
x_ones = torch.ones_like(x_data) #x_data 속성(shape, dtype) 유지 // 이 외에도 cpu, gpu 중 저장된 곳 표시
print(f"ones tensore : \n {x_ones} \n") #즉, x_data의 shape와 dtype은 유지하고 ones이니 요소를 1로 채움

x_rand = torch.rand_like(x_data, dtype=torch.float) #x_data 속성 덮어쓰기 -> shape는 유지, dtype만 변경
print(f"random tensor: \n {x_rand} \n") #random 값을 x_data의 shape인 2x2에 맞게 채움

# 기본 연산
if torch.cuda.is_available():
    print("\n[available] 사용 가능")
    tensor = tensor.to("cuda") #GPU 사용 가능하면 변경

tensor = torch.ones(4,4)
print(f"\nLast Col: {tensor[..., -1]}") #...(엘립시스) -> 모든 이전 차원 // 4x4는 행과 열로 구성된 2차원 배열 -> 1차원 값 반환
                                    #즉, 2차원 값은 1차원 값을 / 3차원 값은 2차원 값을 반환 ?

# 텐서 병합
tensor[:,2] = 0 #모든 행의 3번째 열을 0으로 바꿈
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 텐서 연산
y1 = tensor @ tensor.T #@ = 행렬곱 // a.T = a의 전치행렬
y2 = tensor.matmul(tensor.T) #matmul = @ // y2 == y1

y3 = torch.rand_like(y1) #y1의 속성을 y3에 지정 // 속성 -> shape, dtype, device 저장위치
torch.matmul(tensor, tensor.T, out=y3) #tensor와 tensor.T의 행렬곱 한 값을 out=y3에 저장

# 텐서 요소별 곱
z1 = tensor * tensor #각 요소를 곱함
z2 = tensor.matmul(tensor) #y2 == y1

agg = tensor.sum()
print(f"\nagg = {agg}, agg type: {type(agg)}") 
agg_item = agg.item()
print(f"agg_item: {agg_item}, agg_item type: {type(agg_item)}")
    #출력되는 값은 동일하나, type이 달라 이후 연산에서 오류가 발생할 수 있음

# Numpy 변환 (Bridge)
# CPU상의 Tensor와 Numpy 배열은 메모리 공간 공유 -> 하나 변경 시 다른 하나 자동 변경
t = torch.ones(5)
print(f"\nt: {t}") #tensor 타입
n = t.numpy()
print(f"n: {n}") #numpy 타입

t.add_(1) #t.add()와 t.add_()의 차이는 ?
print(f"\nt: {t}") #CPU에서 메모리 공유하기에, t가 변경하면 n도 변경됨
print(f"n: {n}")

# Numpy to Tensor
n = np.ones(5)
t=torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"\n<Numpy To Tensor> \nt: {t}") 
print(f"n: {n}")



# !최종정리
# 1) Tensor의 속성에는 shape(행,열) / dtype(데이터타입) / device(저장위치) 가 있다.
    #1-1) device는 cpu와 gpu로 구성되어 있다.
    #1-2) torch.cuda.is_availabe() 을 통해 gpu가 동작 가능한 지 확인 가능하다.
    #1-3) cpu 상의 Tensor와 Numpy는 서로 메모리를 공유한다.

# 2) torch.~_like(...) 를 통해 인수의 shape와 dtype을 적용할 수 있다.

# 3) tensor.sum() 을 통해 각 요소의 합을 구할 수 있다.
    #3-1) 이때, tensor.sum()의 dtype은 torch.Tensor이다.
    #3-2) 타입을 변경하고 싶으면 (tensor.sum()).item() 을 통해 float32 타입으로 변환 가능하다.

# 4) Numpy는 Tensor로, Tensor는 Numpy로 변환 가능하다. 