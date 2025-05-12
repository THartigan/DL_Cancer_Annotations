import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
crime_dir = os.path.abspath(os.path.join(script_path, ".."))
sys.path.insert(0, crime_dir)

import torch
import numpy as np
save_directory = "/local/scratch/Data/TROPHY/numpy/"
from Processing.Sample import Sample
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
import torch.optim as optimise
from torch.optim.lr_scheduler import StepLR

class TPL(nn.Module):
    def __init__(self):
        super(TPL, self).__init__()
    
    def forward(self, a, b, c, x):
        Px = c+(1-c)/(1+torch.exp(-a*(x-b)))
        return Px
    
#Inputs is n x 1 x 801
#Outputs is n x 4

class Embedding(nn.Module):
    # Take whole wavenumber sequence and split in to n flattened parts with a desired patch size ~25
    # project into embedding space of dimension D by a learnable lineary projection
    def __init__(self, input_channels, input_length, kernel_size, stride, embedding_dim):
        super(Embedding, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.embedding_dim = embedding_dim
        
        self.linear = nn.Linear(kernel_size, embedding_dim)

    def forward(self, x):
        b, c, l = x.shape
        # x has shape (n, g, 801)
        x = torch.split(x, self.input_channels, dim=1)
        x = torch.stack(x, dim=2)
        x = x.reshape(b, -1)
        x = x.unfold(1, self.kernel_size, self.stride)
    
        x = self.linear(x) # (512, 792, 80)
        return x


# class LearnablePositionalEncoding(nn.Module):
#     # Learn the positional encoding for each patch

# class MultiHeadAttention(nn.Module):


# class FeedForward(nn.Module):

class ViT(nn.Module):
    name = "ViT"

    def __init__(self, input_channels, input_length, embed_kernel_size, embed_stride, embedding_dims, leak_rate, dropout_rate):
        super(ViT, self).__init__()
        self.embedding = Embedding(input_channels, input_length, embed_kernel_size, embed_stride, embedding_dims)

    def forward(self, x):
        
        x = self.embedding(x)
        
        return x