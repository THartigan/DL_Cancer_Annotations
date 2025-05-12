import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
import torch
import numpy as np
from Sample import Sample
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
import torch.optim as optimise
from torch.optim.lr_scheduler import StepLR
from Core_CNN import Core_CNN

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, prop_to_use: float) -> (torch.utils.data.DataLoader, torch.torch.utils.data.DataLoader):
        save_directory = "/local/scratch/Data/TROPHY/numpy/"
        
        samples = np.load(save_directory + "samples.npy", allow_pickle = True)
        normalised_data = np.load(save_directory + "sigmoid_normalised_data.npy")
        classifications = np.load(save_directory + "classification.npy")
        
        print(f"Loaded {len(samples)} samples")
        print(f"Training set final index: {samples[77].end_index}")
        cutoff_index = samples[77].end_index
        
        train_data = torch.tensor(normalised_data[:cutoff_index], dtype=torch.float32).cpu()
        test_data = torch.tensor(normalised_data[cutoff_index:], dtype=torch.float32).cpu()
        train_classes = torch.tensor(classifications[:cutoff_index] - 1, dtype=torch.long).cpu()
        test_classes = torch.tensor(classifications[cutoff_index:] - 1, dtype=torch.long).cpu()
        
        print(f"Reducing to {prop_to_use * 100}% of loaded data")
        num_train = int(len(train_data)*prop_to_use)
        num_test = int(len(test_data)*prop_to_use)
        print(f"Keeping {num_train} for training")
        print(f"Keeping {num_test} for testing")
        keep_train = torch.randperm(len(train_data))[:num_train]
        keep_test = torch.randperm(len(test_data))[:num_test]
        train_data = train_data[keep_train]
        test_data = test_data[keep_test]
        train_classes = train_classes[keep_train]
        test_classes = test_classes[keep_test]
        
        train_dataset = TensorDataset(train_data, train_classes)
        test_dataset = TensorDataset(test_data, test_classes)
        train_loader = DataLoader(train_dataset, batch_size=4608, shuffle=True, num_workers=14, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=4608, shuffle=True, num_workers=14, pin_memory=True)

        train_size = len(train_dataset)
        test_size = len(test_dataset)

        return train_loader, test_loader, train_size, test_size

    def train(self, prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, **kwargs):
        train_loader, test_loader, train_size, test_size = self.load_data(prop_to_use)
        if model_class == Core_CNN:
            # Fixed params
            apriori_relevance = kwargs.get("apriori_relevance", None).to(self.device)
            input_channels = kwargs['input_channels']

            # Hyperperams
            leak_rate = kwargs['leak_rate']
            dropout_rate = kwargs['dropout_rate']
            
            model = Core_CNN(apriori_relevance, leak_rate, dropout_rate, input_channels).to(self.device)
            return self.propagate_model(model, train_loader, test_loader, train_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights.to(self.device)) 

    def propagate_model(self, model, train_loader, test_loader, train_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights) -> float:
        criterion = nn.CrossEntropyLoss(weight=entropy_weights)
        optimiser = optimise.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)

        best_test_accuracy_score = 1000000
        best_test_model_name = ""
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            total_train_correct = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                num_X_samples = len(X_batch)
                X_batch = X_batch.view(num_X_samples, 1, 801)
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
        
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
                # Calculate training stats
                total_train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train_correct += (predicted == Y_batch).sum().item()
            
            # print("Train")
            # print(predicted[:5])
            # print(Y_batch[:5])
        
            model.eval()
            total_eval_loss = 0.0
            total_eval_correct = 0
            total_eval_correct_by_class = np.array([0,0,0,0])
            total_eval_by_class = np.array([0,0,0,0])
            with torch.no_grad():
                for X_batch, Y_batch in test_loader:
                    X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                    num_X_samples = len(X_batch)
                    X_batch = X_batch.view(num_X_samples, 1, 801)
        
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    
                    # Calculate training stats
                    total_eval_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_eval_correct += (predicted == Y_batch).sum().item()
                    for i in range(0,4):
                        correct_mask = (predicted == Y_batch) & (Y_batch == i)
                        total_eval_correct_by_class[i] = correct_mask.sum().item()

                        in_category_mask = Y_batch == i
                        total_eval_by_class[i] = in_category_mask.sum().item()
                # print("Test")
                # print(predicted[:50])
                # print(Y_batch[:50])
        
            scheduler.step()
            avg_train_loss = total_train_loss / train_size
            train_accuracy = total_train_correct / train_size
            
            avg_test_loss = total_eval_loss / test_size
            test_accuracy = total_eval_correct / test_size
        
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {100*train_accuracy:.2f}%, Val Loss: {avg_test_loss:.4f}, Test Accuracy: {100*test_accuracy:.2f}% learning rate: {scheduler.get_last_lr()[0]:.2e}")

            accuracy_by_class = total_eval_correct_by_class / total_eval_by_class
            accuracy_by_class[~np.isfinite(accuracy_by_class)] = 0
            # print(total_eval_correct_by_class)
            # print(total_eval_by_class)
            print(f"Accuracy by class: {accuracy_by_class}")

            accuracy_score = 0
            epsilon =  1E-5
            for accuracy in accuracy_by_class:
                accuracy_score += 1/(accuracy+epsilon)
                
            #if accuracy_score < best_test_accuracy_score:
            best_test_accuracy_score = accuracy_score
            print(f"!!New max test accuracy score {accuracy_score}")
            accuracy_score_string = f"{accuracy_score:.2f}"
        
        
        
            
            torch.save({
                "model": model.name,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_accuracy_score": accuracy_score,
                "test_accuracy_by_category": accuracy_by_class},
                "/local/scratch/Data/TROPHY/Models/" + accuracy_score_string + model.name+".pth")
            
            best_test_accuracy_score = accuracy_score
            
            if best_test_model_name != "":
                remove_file_path = "/local/scratch/Data/TROPHY/Models/" + best_test_model_name
                
                if os.path.exists(remove_file_path):
                    os.remove(remove_file_path)
                else:
                    print("Previous best file not found")
                    
            best_test_model_name = accuracy_score_string + model.name+".pth"

            if best_test_accuracy_score > 5000:
                print(f"Skipping the rest of training, best: {best_test_accuracy_score}")
                return best_test_accuracy_score
                break

                
        print(f"Training complete, best: {best_test_accuracy_score}")
        return best_test_accuracy_score
        
        # Best:  75.87% 
                    