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
print(imgs[0]) # just making sure it looks ok

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        print("hi")
        # self.cnn_block = nn.Sequential(
        #     nn.Conv2d(1)
        # )

