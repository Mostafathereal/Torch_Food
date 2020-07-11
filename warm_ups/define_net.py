import cv2

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):

        ## inheriting from nn module
        super(Net, self).__init__()

        # 1 in channel, 6  out channels, 3x3 kernel 
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)


        ## image will be 6x6
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        # first block: x --> conve1 --> relu --> max_ppol
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        # second block
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # reshaping the tensor to #batches by #features in 1 sample
        x = x.view(-1, self.num_flat_features(x))

        # first fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # softmax or whatever output activation function will be applied later if necessary
        x = self.fc3(x)

        return x

    # helper  method for flattening the last conv layer into a fc layer
    def num_flat_features(self, x):
        size = x.size()[1:] # exclude batch dimension
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
for i in params:
    print(i.size())

inpt = torch.randn(1, 1, 32, 32)

## this will invoke the "forward" method we difned earlier
out = net(inpt)
print(out)

target = torch.randn(10) # youre 'label'
target = target.view(1, -1) ## make dummy target same shape as output 

criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)

# # now we follow `loss` backward using .grad_fn attribute 
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()
loss.backward()
print("conv1 bias grad: ", net.conv1.bias.grad)
print("conv1 weight grad: ", net.conv1.weight.grad)

#stochastic GD
lr = 0.001
for p in net.parameters():
    p.data.sub_(p.grad.data * lr)

# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print (name, param.size())

