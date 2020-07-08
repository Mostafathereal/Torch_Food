import torch
import time

dtype = torch.float
device = torch.device("cuda:0")

N = 32
D_in, D_out = 1000, 10
H = 100

x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

lr = 1e-6

start = time.time()

for epoch in range(500):
    z1 = x.mm(w1)
    a1 = z1.clamp(min = 0)
    z2 = a1.mm(w2)
    yhat = z2

    loss = (yhat - y).pow(2).sum().item()

    print(epoch, loss)
    dz2 = dyhat = (2* (yhat - y))
    dw2 = a1.t().mm(dz2)
    dz1 = dz2.mm(w2.t())
    dz1[z1<0] = 0
    dw1 = x.t().mm(dz1) 

    w1 -= lr * dw1
    w2 -= lr * dw2

print("time = ", time.time() - start)