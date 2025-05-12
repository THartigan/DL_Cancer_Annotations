from Processing.Trainer import Trainer
import torch
import numpy as np
from Models.Core_CNN import Core_CNN
from Models.Core_CNN_TPL import Core_CNN_TPL
from Models.SCNN_TPL import SCNN_TPL
from Models.XGBoost import XGB_Train
from Models.Core_CNN_TPL2 import Core_CNN_TPL2
from Models.SCNN_TPL3 import SCNN_TPL3
from Models.SCNN_TPL2 import SCNN_TPL2
from Models.ViT import ViT
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

class CombinationModel:
    def __init__(self, model_class, component_models, num_classes, combination_mode="vote", batch_size=50000):
        model_class_dict = {
            "SCNN_TPL": SCNN_TPL,
            "Core_CNN_TPL": Core_CNN_TPL,
            "Core_CNN_TPL2": Core_CNN_TPL2,
            "SCNN_TPL2": SCNN_TPL2,
            "SCNN_TPL3": SCNN_TPL3,
            "Core_CNN": Core_CNN,
            "XGB_Train": XGB_Train,
            "ViT": ViT
        }
        self.model_class = model_class_dict[model_class]
        self.component_models = component_models
        self.num_classes = num_classes
        # Could also implement an "average" mode to try to get a sense of the confidence, but would require relabelling classes to put them in serverity order
        self.combination_mode = combination_mode
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apriori_relevance = torch.ones(801).to(self.device)
        self.loaded_models = self.preload_models()
        

    def predict(self, data: np.ndarray):
        torch_data = TensorDataset(torch.tensor(data).float().to(self.device))
        print(data.shape)
        data_loader = DataLoader(torch_data, batch_size=self.batch_size, shuffle=False)
    
        all_predictions = []
        with torch.no_grad():
            batch_count = 0
            for X_batch in data_loader:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Batch {batch_count} of {len(data_loader)}")
                x_data = X_batch[0].unsqueeze(1)
                batch_predictions = []
                for model in self.loaded_models:
                    pred = model(x_data)
                    _, predicted = torch.max(pred, 1)
                    batch_predictions.append(predicted.cpu().numpy())

                if not all_predictions:
                    all_predictions = [pred for pred in batch_predictions]
                else:
                    for i, pred in enumerate(batch_predictions):
                        all_predictions[i] = np.concatenate([all_predictions[i], pred])
                
                # print(np.shape(all_predictions))
        all_predictions = np.array(all_predictions)

        
        if self.combination_mode == "vote":
            ensemble_predictions = np.zeros((len(all_predictions[0]),), dtype=int)
            for i in range(len(all_predictions[0])):
                votes = all_predictions[:,i]
                most_common = Counter(votes).most_common(1)
                # print(most_common)
                ensemble_predictions[i] = most_common[0][0]
            return ensemble_predictions
        
        if self.combination_mode == "vote_probs":
            ensemble_predictions = np.zeros((len(all_predictions[0]), self.num_classes), dtype=float)
            for i in range(len(all_predictions[0])):
                votes = all_predictions[:,i]
                values, counts = np.unique(votes, return_counts=True)
                for value in values:
                    ensemble_predictions[i, value] = counts[values == value]/len(votes)

                # print(values)
                probs = counts / len(votes)
                # print(probs)
            return ensemble_predictions

        
        


    def preload_models(self):
        print(f"Preloading {len(self.component_models)} models...")
        models = []
        for model_params in self.component_models:
            model = self.model_class(self.apriori_relevance, 
                                model_params['config']['leak_rate'], 
                                model_params['config']['dropout_rate'], 1)
            model = model.to(self.device)
            model.load_state_dict(torch.load(model_params['model_path'])['model_state_dict'])
            model.eval()
            models.append(model)
            print(f"Loaded model: {model_params['model_path']}")
        return models
        