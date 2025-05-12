import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
save_directory = "/local/scratch/Data/TROPHY/numpy/"
from Sample import Sample
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
import torch.optim as optimise
from torch.optim.lr_scheduler import StepLR
# samples = np.load(save_directory + "samples.npy", allow_pickle = True)
# normalised_data = np.load(save_directory + "sigmoid_normalised_data.npy")
# classifications = np.load(save_directory + "classification.npy")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# print(len(samples))
# print(samples[77].end_index)
# cutoff_index = samples[77].end_index


# train_data = torch.tensor(normalised_data[:cutoff_index], dtype=torch.float32).cpu()
# test_data = torch.tensor(normalised_data[cutoff_index:], dtype=torch.float32).cpu()
# train_classes = torch.tensor(classifications[:cutoff_index] - 1, dtype=torch.long).cpu()
# test_classes = torch.tensor(classifications[cutoff_index:] - 1, dtype=torch.long).cpu()

# prop_to_use = 1 #len(normalised_data)
# num_train = int(len(train_data)*prop_to_use)
# num_test = int(len(test_data)*prop_to_use)
# print(f"Keeping {num_train} for training")
# print(f"Keeping {num_test} for testing")
# keep_train = torch.randperm(len(train_data))[:num_train]
# keep_test = torch.randperm(len(test_data))[:num_test]
# train_data = train_data[keep_train]
# test_data = test_data[keep_test]
# train_classes = train_classes[keep_train]
# test_classes = test_classes[keep_test]

# print(train_classes[10])
# train_dataset = TensorDataset(train_data, train_classes)
# test_dataset = TensorDataset(test_data, test_classes)
# train_loader = DataLoader(train_dataset, batch_size=4608, shuffle=True, num_workers=14, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=4608, shuffle=True, num_workers=14, pin_memory=True)

class Core_CNN(nn.Module):
    def __init__(self, apriori_relevance, leak_rate, dropout_rate, input_channels):
        super(Core_CNN, self).__init__()
        # Multi-scale Assessing Layers
        self.msal8 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8) # Output shape (8,794) 
        self.msal25 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25) # Output shape (8, 777)
        self.msal50 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=50) #Output shape (8, 752)
        # An array of length 801 denoting an apriori understanding of the imporance of each channel
        self.apriori_relevace = apriori_relevance

        # Core CNN Layers
        self.reLU = nn.LeakyReLU(leak_rate)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.core1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=15, stride=5)
        self.core2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, padding="same")
        self.core3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9, padding="same")
        self.core4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding="same")
        self.core5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding="same")
        self.core6 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        self.core7 = nn.Linear(10048, 32) #25152
        self.core8 = nn.Linear(32,32)
        self.core9 = nn.Linear(32,16)
        self.core10 = nn.Linear(16,16)

        # Output Layers
        self.out1 = nn.Linear(16,4)
        
    
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
        # Core CNN Layers
        x_core = self.core1(x_core)
        x_core = self.bn1(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)
        x_core = self.core2(x_core)
        x_core = self.bn2(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.core3(x_core)
        x_core = self.bn2(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)
        x_core = self.core4(x_core)
        x_core = self.bn3(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.core5(x_core)
        x_core = self.bn3(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)
        x_core = self.core6(x_core)
        x_core = self.bn4(x_core)
        x_core = self.dropout(x_core)
        x_core = x_core.view(len(x_core), -1)
        # print("At linear")
        x_core = self.core7(x_core)
        x_core = self.bn3(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.core8(x_core)
        x_core = self.bn3(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)
        x_core = self.core9(x_core)
        x_core = self.bn2(x_core)
        x_core = self.dropout(x_core)
        x_core = self.tanh(x_core)
        x_core = self.core10(x_core)
        x_core = self.bn2(x_core)
        x_core = self.dropout(x_core)
        x_core = self.reLU(x_core)
        # print(x_core.shape)


        # Output Layers
        x_out = self.out1(x_core)
        
        # print(x_out)

        
        # print(x_core.shape)
        # x = x.view(x.size(0), -1)
        # x = self.output_layer(x)
        return x_out


# apriori_relevance = torch.ones(801).to(device)
# model = Core_CNN(apriori_relevance).to(device)
# criterion = nn.CrossEntropyLoss()
# optimiser = optimise.AdamW(model.parameters(), lr=1E-4, weight_decay=0.003)
# scheduler = StepLR(optimiser, step_size=5, gamma=0.5)

# print(device)
# epochs = 10000
# for epoch in range(epochs):
#     model.train()
#     total_train_loss = 0.0
#     total_train_correct = 0
    
#     for X_batch, Y_batch in train_loader:
#         X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
#         num_X_samples = len(X_batch)
#         X_batch = X_batch.view(num_X_samples, 1, 801)
        
#         outputs = model(X_batch)
#         loss = criterion(outputs, Y_batch)

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         # Calculate training stats
#         total_train_loss += loss.item() * X_batch.size(0)
#         _, predicted = torch.max(outputs, 1)
#         total_train_correct += (predicted == Y_batch).sum().item()
#         # print(f"p: {predicted}")
#         # print(f"b: {Y_batch}")
#     print("Train")
#     print(predicted[:100])
#     print(Y_batch[:100])

#     model.eval()
#     total_eval_loss = 0.0
#     total_eval_correct = 0
#     with torch.no_grad():
#         for X_batch, Y_batch in test_loader:
#             X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
#             num_X_samples = len(X_batch)
#             X_batch = X_batch.view(num_X_samples, 1, 801)

            
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
            
#             # Calculate training stats
#             total_eval_loss += loss.item() * X_batch.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total_eval_correct += (predicted == Y_batch).sum().item()
#         print("Test")
#         print(predicted[:100])
#         print(Y_batch[:100])

#     scheduler.step()
#     avg_train_loss = total_train_loss / len(train_dataset)
#     train_accuracy = 100* total_train_correct / len(train_dataset)
#     avg_val_loss = total_eval_loss / len(test_dataset)
#     val_accuracy = 100* total_eval_correct / len(test_dataset)

#     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% learning rate: {scheduler.get_last_lr()[0]:.6f}")

# print("Training complete")

# # Best:  75.87% 
