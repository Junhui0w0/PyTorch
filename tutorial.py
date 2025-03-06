import torch
import numpy as np

#예제 데이터로부터 직접 Tensor 생성
data = [[1,2], [3,4]]
x_data = torch.tensor(data) #data의 자료형은 알아서 유추됨

np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

x_ones = torch.ones_like(x_data) #x_data 속성(shape, dtype) 유지
print(f"ones tensore : \n {x_ones} \n") #즉, x_data의 shape와 dtype은 유지하고 ones이니 요소를 1로 채움

x_rand = torch.rand_like(x_data, dtype=torch.float) #x_data 속성 덮어쓰기 -> shape는 유지, dtype만 변경
print(f"random tensor: \n {x_rand} \n") #random 값을 x_data의 shape인 2x2에 맞게 채움

shape = (2,3,) #맨 뒤에 , 왜 추가하는 거임? ->
rand_tensor = torch.rand(shape)
print(f"Random Tensor (w. shape): \n {rand_tensor} \n")

