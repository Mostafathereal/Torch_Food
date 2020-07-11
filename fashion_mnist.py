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


#always look at your datasets
print(trainset)
print(testset)

batch1 = iter(testloader)
imgs, labels = batch1.__next__()
print(labels)
print(imgs.size()) # just making sure it looks ok

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        ## note that vgg always does same padding on convolutions
        ## dec img size by pooling and inc channels using kernels
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU()
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 265, 3, padding = 1),
            nn.ReLU()
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU()
            nn.MaxPool2d(2, 2)

            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU()
        )

        self.fc_block = nn.Sequential(
            nn.Linear()
        )

    def num_flat_features(self, x):
        size = x.size()[1:] # exclude batch dimension
        num_features = 1
        for i in size:
            num_features *= i
        return num_features
