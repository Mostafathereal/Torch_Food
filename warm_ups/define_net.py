
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.module):

    def __init__(self):

        ## inheriting from nn module
        super(Net, self).__init()

        # 1 in channel, 6  out channels, 3x3 kernel 
        self.conv1 = nn.conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)


        ## image will be 6x6
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

