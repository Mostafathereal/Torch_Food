import torch
import time

dtype = torch.float
device = torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10


x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

lr = 1e-6

start = time.time()

for epoch in range(500):

    # this is a 1 line'r cuz we dont need all the intermediate variables 
    # because of the fact that we are'nt implementing back prop ourselves
    yhat = x.mm(w1).clamp(min = 0).mm(w2)

    loss = (yhat - y).pow(2).sum()

    # print(epoch, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
print("time = ", time.time() - start)