# PyTorch는 GPU로 연산 속도를 향상시킬 수 있는 N차원 Tensor를 사용
# 신경망을 구성 및 학습하는 과정에서 자동 미분(Autograd) 수행 가능

#====[Tensor]====#
import torch
import math

dtype = torch.float
device = torch.device("cuda:0")

#랜덤 데이터와 출력
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

#랜덤 가중치 초기화
a=torch.randn((), device=device, dtype=dtype)
b=torch.randn((), device=device, dtype=dtype)
c=torch.randn((), device=device, dtype=dtype)
d=torch.randn((), device=device, dtype=dtype)

lr = 1e-6
for t in range(2000):
    #순전파: 예측값 y 계산
    y_pred = a+ b*x + c * x**2 + d * x ** 3

    #loss 계산
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss) #t=시행횟수 // loss=손실값

    #loss에 따른 a,b,c,d의 그래디언트 계산 및 역전파
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    #가중치 갱신
    a -= lr * grad_a
    b -= lr * grad_b
    c -= lr * grad_c
    d -= lr * grad_d

print(f"[Tensor] Res: y = {a.item()} + {b.item()}x + {c.item()}x^2 + {d.item()}x^3\n\n")


#====[Autograd - 자동미분]====#
x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y=torch.sin(x)

a=torch.randn((), dtype=dtype, requires_grad=True)
b=torch.randn((), dtype=dtype, requires_grad=True)
c=torch.randn((), dtype=dtype, requires_grad=True)
d=torch.randn((), dtype=dtype, requires_grad=True) #텐서들의 변화도 계산 (requires_grad = True)

for t in range(2000):
    y_pred = a+ b*x + c * x**2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item()) #loss.item() == 손실값이 가지고 있는 스칼라 값

    loss.backward() #역전파 수행 == requires_grad=True 값을 갖는 모든 텐서들에 대한 손실의 변화도 계산
                    #-> a.grad, b.grad, c.grad, d.grad는 각 a b c d에 대한 손실의 변화도 갖는 텐서

    with torch.no_grad(): #추적 방지 // 매개변수 고정
        a-= lr * a.grad
        b-= lr * b.grad
        c-= lr * c.grad
        d-= lr * d.grad

        #가중치 갱신 후 0으로 초기화
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"[Autograd] Res: y = {a.item()} + {b.item()}x + {c.item()}x^2 + {d.item()}x^3\n\n")

#========================================================================#
# 1-1. PyTorch는 GPU를 통해 연산 속도를 높이기 위해 N차원 Tensor를 사용한다. 
# 1-2. 신경망 구성 및 학습하는 과정에서 자동미분 (Autograd) 과정을 수행할 수 있다.

# 2-1. PyTorch에서 backward() 메소드를 통해 역전파 (손실함수의 기울기) 를 구할 수 있다.
# 2-1-1. 손실함수 기울기 탐색 이유 == 파라미터 최적화
# 2-2. 계산된 기울기는 각 매개변수의 'grad' 속성에 저장된다.
# 2-3. requires_grad=True 로 지정된 텐서들의 손실도를 계산한다. 

# 3-1. Autograd == 순전파와 역전파 과정 기록
# 3-2. backward == Autograd의 기능 중 역전파 수행
#========================================================================#