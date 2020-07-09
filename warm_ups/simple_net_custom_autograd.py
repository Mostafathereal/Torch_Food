import torch

class MyRelu(torch.autograd.Function):

    def forward(ctx, input)