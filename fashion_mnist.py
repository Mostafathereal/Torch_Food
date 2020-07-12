import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size= 8, shuffle = False)

device = torch.device("cuda")
print(device)

#always look at your datasets
print(trainset)
print(testset)

# batch1 = iter(testloader)
# imgs, labels = batch1.__next__()
# print(labels)
# print(imgs.size()) # just making sure it looks ok

class vgg16(nn.Module):

    def __init__(self):
        super(vgg16, self).__init__()

        ## note that vgg always does same padding on convolutions
        ## dec img size by pooling and inc channels using kernels
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
            # out = 6x6 img
        )

        self.fc_block = nn.Sequential(
            # 6x6x512 = 18432
            nn.Linear(18432, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            # nn.Softmax(dim = 1) 
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:] # exclude batch dimension
    #     num_features = 1
    #     for i in size:
    #         num_features *= i
    #     return num_features

net = vgg16().to(device)
loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr = 0.0001)

# # output looks ok
# dat = iter(testloader).__next__()[0].to(device)
# print(dat.size())
# out = net(dat)
# print(out)
# dat = iter(testloader).__next__()[0].to(device)
# print(dat.size())
# out = net(dat)
# print(out)
# print("hehehehehehhehehehehe")
# print(out.data)
# _, preds = torch.max(out.data, 1)
# print("heheheheheh22222")
# print(preds)

# calculate accuracy using GPU
def evalaluate(dataLoader):
    total = 0
    correct = 0
    net.eval() # put model in eval mode

    for data in dataLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs.data, 1)
        total += labels.size()
        correct += (predictions == labels).sum().item()
    return (correct/total) * 100




