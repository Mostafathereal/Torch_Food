## just a review of how to build a NN using NumPy only
import numpy as np

# dimensions
N = 32
D_in, D_out = 1000, 10
H = 100

#layers
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

#learning ra
# te
lr = 1e-6

for epoch in range(300):

    ###   Forward prop   ###
    # sum of prod
    z1 = x.dot(w1)
    #relu
    a1 = np.maximum(z1, 0)

    z2 = a1.dot(w2)
    ## no activation here
    # a2 = yhat
    yhat = a1.dot(w2)
    # no activation for out

    loss = np.square(yhat - y).sum()
    # print(yhat)
    print(epoch, loss)

    ## derivative of mean square error:
    # ((yhat - y)^2)' = 2(yhat - y)

    ###   Back Prp   ###
    dz2 = dyhat = (2.0 * (yhat - y))
    dw2 = np.dot((a1.T), dz2)
    dz1 = dz2.dot(w2.T)
    dz1[z1 < 0] = 0
    dw1 = x.T.dot(dz1)

    w1 -= lr * dw1
    w2 -= lr * dw2







