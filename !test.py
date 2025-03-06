import torch
print("Using torch version", torch.__version__)
torch.manual_seed(42) #동일 결과 방지

#예제 데이터로 실습
x= torch.tensor([[1., -1.],
              [2., 3.]], requires_grad=True) #requires_grad=Tensor에 대한 미분 과정 기록 여부
y = x.pow(2).sum() #y=x^2
y.backward() #자동미분 -> dy/dx = 2x
print(f"\ndy/dx = {x.grad}")

x = torch.tensor(2.0, requires_grad=True)
y = 2 * x**2 + 5 #y=2x^2 + 5
y.backward() #역전파 
print(f"\n수식 y를 x로 미분한 값: {x.grad}") #y'=4x

