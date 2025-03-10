# 신경망에서는 '역전파' 라고 불리는 알고리즘을 주로 사용한다.
    # 역전파 == 오차값을 역방향으로 전파하여 가중치를 조절하는 것
# 해당 알고리즘의 가중치는 Gradient에 따라 조정된다.
# 이는 PyTorch의 AutoGrad를 통해 쉽게 계산할 수 있다.

import torch
x = torch.ones(5) #[1,1,1,1,1] 입력값 
print(f"x = {x}\n")

y = torch.zeros(3) #[0,0,0] 출력값

w = torch.randn(5,3, requires_grad=True) # 최적화가 필요한 매개변수 // 5행 3열
print(f"w = {w}\n")

b = torch.randn(3, requires_grad=True) # 최적화가 필요한 매개변수 -> requires_grad=True
    #b.requires_grad_(True)를 통해 이후에도 적용 가능

z = torch.matmul(x,w) + b #x와 w 행렬곱
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f" Gradient function for z = {z.grad_fn}\n")
print(f"Gradient function for loss = {loss.grad_fn}\n")


# Gradient 계산
# 매개변수의 가중치 최적화 == 매개변수에 대한 손실함수의 도함수
loss.backward() #성능 상 backward()는 동일 그래프에서 1회 수행 가능
    #다만, 여러번 수행 할 경우 retrain_graph=True 지정

print(f"w.grad = {w.grad}\n")
print(f"y.grad = {y.grad}\n")


# 변화도 추적 중단
# requires_grad=True인 텐서들은 연산 기록 추적 및 변화도 계산 지원
# 순전파 연산만 수행한다 == torch.no_grad() 블록으로 둘러싸 추적 중단 가능
z = torch.matmul(x, w) + b
print(f"z.requires_grad = {z.requires_grad}\n")

with torch.no_grad(): #추적 중단 // detach() 메소드를 통해 동일 결과 수행 가능
    z = torch.matmul(x, w) + b
print(f"z.requires_grad = {z.requires_grad}")

# 변화도 추적을 중단해야 하는 이유?
# 1) 신경망 일부 매개변수를 고정된 매개변수로 표시할 때
# 2) 순전파 단계만 수행할 때, 연산 속도 향상


# 순전파 단계의 autograd는 아래 2가지 작업을 동시 수행함.
# 1) 결과 텐서 계산
# 2) 방향성 비순환 그래프 (DAG)에 Gradient Function 유지


# 역전파는 DAG의 root에서 backward()가 호출될 때 시작
# 1) 각 .grad_fn 으로부터 변화도 계산
# 2) 각 텐서의 .grad 속성에 계산 결과 축적
# 3) 연쇄 법칙을 통해 모든 Leaf Tensor 까지 전파
