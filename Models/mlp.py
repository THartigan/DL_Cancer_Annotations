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
samples = np.load(save_directory + "samples.npy", allow_pickle = True)
normalised_data = np.load(save_directory + "sigmoid_normalised_data.npy")
classifications = np.load(save_directory + "classification.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(len(samples))
print(samples[77].end_index)
cutoff_index = samples[77].end_index

train_data = torch.tensor(normalised_data[:cutoff_index], dtype=torch.float32).cpu() 
test_data = torch.tensor(normalised_data[cutoff_index:], dtype=torch.float32).cpu()
train_classes = torch.tensor(classifications[:cutoff_index] - 1, dtype=torch.long).cpu()
test_classes = torch.tensor(classifications[cutoff_index:] - 1, dtype=torch.long).cpu()

print(train_classes[10])
train_dataset = TensorDataset(train_data, train_classes)
test_dataset = TensorDataset(test_data, test_classes)
train_loader = DataLoader(train_dataset, batch_size=9600, shuffle=True, num_workers=14, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=9600, num_workers=14, pin_memory=True)

class MLPModel(nn.Module):
    def __init__(self, input_size, l1_size, l2_size, l3_size, l4_size, l5_size, output_size):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, l1_size)
        self.layer2 = nn.Linear(l1_size, l2_size)
        self.layer3 = nn.Linear(l2_size, l3_size)
        self.layer4 = nn.Linear(l3_size, l4_size)
        self.layer5 = nn.Linear(l4_size, l5_size)
        self.layer6 = nn.Linear(l5_size, output_size)
        self.reLU = nn.ReLU()

    def forward(self, x):
        x = self.reLU(self.layer1(x))
        x = self.reLU(self.layer2(x))
        x = self.reLU(self.layer3(x))
        x = self.reLU(self.layer4(x))
        x = self.reLU(self.layer5(x))
        x = self.layer6(x)
        return x

model = MLPModel(801, 50, 50, 30, 30, 10, 4).to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optimise.Adam(model.parameters(), lr=0.0001)

print(device)
epochs = 400
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
        
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Calculate training stats
        total_train_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train_correct += (predicted == Y_batch).sum().item()

    model.eval()
    total_eval_loss = 0.0
    total_eval_correct = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            # Calculate training stats
            total_eval_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total_eval_correct += (predicted == Y_batch).sum().item()

    avg_train_loss = total_train_loss / len(train_dataset)
    train_accuracy = 100* total_train_correct / len(train_dataset)
    avg_val_loss = total_eval_loss / len(test_dataset)
    val_accuracy = 100* total_eval_correct / len(test_dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%") ##learning rate: {scheduler.get_last_lr()[0]:.6f}

print("Training complete")