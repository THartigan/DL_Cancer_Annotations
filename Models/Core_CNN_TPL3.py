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

class Core_CNN_TPL3(nn.Module):
    name = "Core_CNN_TPL3"

    def __init__(self, apriori_relevance, leak_rate, dropout_rate, input_channels):
        super(Core_CNN_TPL3, self).__init__()
        # Multi-scale Assessing Layers
        self.msal8 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8, stride=4) # Output shape (8,794) 
        self.msal25 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25, stride=12) # Output shape (8, 777)
        self.msal50 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=50, stride=25) #Output shape (8, 752)
        # An array of length 801 denoting an apriori understanding of the imporance of each channel
        self.apriori_relevace = apriori_relevance

        # Core CNN Layers
        self.reLU = nn.LeakyReLU(leak_rate)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.core1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=15, stride=7)
        self.core_norm1 = nn.BatchNorm1d(8)

        self.core2_1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, padding="same")
        self.core_norm2_1 = nn.BatchNorm1d(16)
        self.core2_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9, padding="same")
        self.core_norm2_2 = nn.BatchNorm1d(16)

        self.core3_1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding="same")
        self.core_norm3_1 = nn.BatchNorm1d(32)
        self.core3_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding="same")
        self.core_norm3_2 = nn.BatchNorm1d(32)

        self.core4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        self.core_norm4 = nn.BatchNorm1d(64)

        self.core5 = nn.Linear(10048, 32) #25152
        self.core_norm5 = nn.BatchNorm1d(32)

        self.core6 = nn.Linear(32,16)
        self.core_norm6 = nn.BatchNorm1d(16)
        

        ### Output Layers
        # Determine a,b,c parameters for 3 TPL
        self.outa = nn.Linear(16,4)
        self.outb = nn.Linear(16,4)
        self.outc = nn.Linear(16,4)
        self.outx = nn.Linear(16,4)

        # Determine output values for each class
        self.tpl = TPL()


        
        
    
    def forward(self, x):
        # Multi-Scale assessing Layer
        # x_msal8 = self.msal8(x)
        # x_msal8 = self.bn1(x_msal8)
        # x_msal8 = self.dropout(x_msal8)
        # x_msal25 = self.msal25(x)
        # x_msal8 = self.bn1(x_msal25)
        # x_msal8 = self.dropout(x_msal25)
        # x_msal50 = self.msal50(x)
        # x_msal8 = self.bn1(x_msal50)
        # x_msal8 = self.dropout(x_msal50)
        # x_msal_apriori = x * self.apriori_relevace
        # x_msal_apriori = self.bn0(x_msal_apriori)
        # x_msal_apriori = self.dropout(x_msal_apriori)
        
        # # print(x_msal8.shape)
        # # print(x_msal25.shape)
        # # print(x_msal50.shape) 
        # # print(x_msal_apriori.shape)
        
        # x_msal_convs = torch.concat([x_msal8, x_msal25, x_msal50], axis=2)
        # # print(x_msal_convs.shape)
        # x_msal_convs_flat = x_msal_convs.view(x_msal_convs.size(0),-1)
        # x_msal_apriori_flat = x_msal_apriori.view(x_msal_apriori.size(0),-1)
        # x_core = torch.concat([x_msal_convs_flat, x_msal_apriori_flat], axis=1) # (1, 19385)
        # # print(x_core.shape)
        # x_core = x_core.view(len(x_core), 1, 19049)
        # print(x_core[1])

        x_core = x
        ### Core CNN Layers
        ## Convolution Layers
        # First Convolution Layer
        x_core = self.core1(x_core)
        x_core = self.core_norm1(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)

        # First tanh-relu bock
        x_core = self.core2_1(x_core)
        x_core = self.core_norm2_1(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.reLU(x_core)

        # Second tanh-relu block
        x_core = self.core3_1(x_core)
        x_core = self.core_norm3_1(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.reLU(x_core)

        # Fine convolution layer
        x_core = self.core4(x_core)
        x_core = self.core_norm4(x_core)
        x_core = self.dropout(x_core)
        x_core = x_core.view(len(x_core), -1)

        ## Linear Layers
        x_core = self.core5(x_core)
        x_core = self.core_norm5(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.reLU(x_core)

        x_core = self.core6(x_core)
        x_core = self.core_norm6(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.reLU(x_core)
       
        # print(x_core)


        # Output Layers
        a = self.outa(x_core)
        b = self.outb(x_core)
        c = self.outc(x_core)
        x_out = self.outx(x_core)
        x_out = self.tpl(a,b,c,x_out)
        
        # print(x_out)

        
        # print(x_core.shape)
        # x = x.view(x.size(0), -1)
        # x = self.output_layer(x)
        return x_out
