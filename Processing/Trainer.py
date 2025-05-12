import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
crime_dir = os.path.abspath(os.path.join(script_path, ".."))
sys.path.insert(0, crime_dir)

import torch
import numpy as np
import Processing.Sample as Sample
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
import torch.optim as optimise
from torch.optim.lr_scheduler import StepLR
from Models.Core_CNN import Core_CNN
from Models.Core_CNN_TPL import Core_CNN_TPL
from Models.SCNN_TPL import SCNN_TPL
from Models.Core_CNN_TPL2 import Core_CNN_TPL2
from Models.SCNN_TPL2 import SCNN_TPL2
from Models.SCNN_TPL3 import SCNN_TPL3
from Models.ViT import ViT
from Models.XGBoost import XGB_Train
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Processing.misc import calculate_performance_stats_by_class
sys.modules['Sample'] = Sample
from sklearn.model_selection import StratifiedGroupKFold
import wandb
import inspect

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

    # Loading process peaks around 11GB of memory usage
    def load_data(self, prop_to_use: float, batch_size: int, number_of_training_folds, validation_test_folds, training_fold_number: int = -1, val_fold_number: int =1, return_loaders=True, display_group_details=False, randomise=True, return_coords=False) -> tuple[torch.utils.data.DataLoader, torch.torch.utils.data.DataLoader, int, int]:
        print(f"Loading data with {number_of_training_folds} training folds and {validation_test_folds} testing folds")
        if os.path.isdir("/local/scratch/Data/TROPHY/numpy"):
            print("On csvm5")
            save_directory = "/local/scratch/Data/TROPHY/numpy/"
            self.models_path = "/local/scratch/Data/TROPHY/Models/"
        elif os.path.isdir("/local/scratch-3/tjh200/processed_trophy_data"):
            print("On Kiiara")
            save_directory = "/local/scratch-3/tjh200/processed_trophy_data/"
            self.models_path = "/local/scratch-3/tjh200/Models/"
        elif os.path.isdir("/local/scratch/data/TROPHY/numpy"):
            print("On PC")
            save_directory = "/local/scratch/data/TROPHY/numpy/"
            self.models_path = "/local/scratch/data/TROPHY/Models/"
        elif os.path.isdir("/Volumes/T7/scratch/data/TROPHY/numpy"):
            print("On Mac")
            save_directory = "/Volumes/T7/scratch/data/TROPHY/numpy/"
            self.models_path = "/Volumes/T7/scratch/data/TROPHY/Models/"
        else:
            raise Exception("Save directory not found")
        
        samples: list[Sample.Sample] = np.load(save_directory + "samples.npy", allow_pickle = True)
        normalised_data = np.load(save_directory + "data.npy")#"sigmoid_normalised_data.npy")
        normalised_data = normalised_data.T
        classifications = np.load(save_directory + "classification.npy")
        if return_coords:
            coords = np.load(save_directory + "coords.npy").T
            sample_nums = np.load(save_directory + "sample_num.npy")[0]
            print(np.shape(sample_nums))
            print(np.unique(sample_nums))
            if randomise:
                raise Exception("Cannot randomise when returning coordinates")
            
        print(f"Loaded {len(samples)} samples")

        # Stratified Group K Fold Implementation
        groups = []
        for i, sample in enumerate(samples):
            sample_size = sample.end_index - sample.start_index + 1
            indices = np.ones(sample_size) * i
            groups = np.concatenate([groups, indices])

        skf = StratifiedGroupKFold(n_splits = number_of_training_folds)
        training_indices = []
        val_test_indices = []
        match_scores = []

        chosen_training_fold = training_fold_number
        chosen_validation_fold = val_fold_number
        best_val_fold_scores = []
        best_val_folds = []

        # Selecting the best fold for continued use
        for i, (split_train_indices, split_val_test_indices) in enumerate(skf.split(normalised_data, classifications, groups)):
            training_indices.append(split_train_indices)
            # training_ys = classifications[split_train_indices]
            val_test_ys = classifications[split_val_test_indices]

            skf_val_test = StratifiedGroupKFold(n_splits = validation_test_folds)
            val_indices = []
            test_indices = []
            
            for split_val_indices, split_test_indices in skf_val_test.split(normalised_data[split_val_test_indices], val_test_ys, groups[split_val_test_indices]):
                
                val_indices.append(split_val_test_indices[split_val_indices])
                test_indices.append(split_val_test_indices[split_test_indices])
                if display_group_details:
                    train_samples = np.unique(groups[split_train_indices]).astype(int).tolist()
                    train_classes = [int(samples[int(i)].calculate_answers(classifications)[0]) for i in train_samples]
                    train_output = list(zip(train_samples, train_classes))
                    print(f"Train samples and classes: {train_output}")
                    val_samples = np.unique(groups[split_val_test_indices[split_val_indices]]).astype(int).tolist()
                    val_classes = [int(samples[int(i)].calculate_answers(classifications)[0]) for i in val_samples]
                    val_output = list(zip(val_samples, val_classes))
                    print(f"Val samples and classes: {val_output}")
                    test_samples = np.unique(groups[split_val_test_indices[split_test_indices]]).astype(int).tolist()
                    test_classes = [int(samples[int(i)].calculate_answers(classifications)[0]) for i in test_samples]
                    test_output = list(zip(test_samples, test_classes))
                    print(f"Test samples and classes: {test_output}")
                # print(train_classes)
                # val_samples = np.unique(groups[split_val_test_indices[split_val_indices]])
                # test_samples = np.unique(groups[split_val_test_indices[split_test_indices]])
                
                # print(f"Train samples: {np.unique(groups[split_train_indices])}")
                # print(f"Val samples: {np.unique(groups[split_val_test_indices[split_val_indices]])}")
                # print(f"Test samples: {np.unique(groups[split_val_test_indices[split_test_indices]])}")
                # print(np.unique(groups[split_val_test_indices[split_val_indices]]))
                # print(np.unique(groups[split_val_test_indices[split_test_indices]]))
                # train_split_val_ys = classifications[split_val_indices]
                # train_split_test_ys = classifications[split_test_indices]

            
            split_val_test_indices_pairs = [{"val": val_indices[i], "test": test_indices[i]} for i in range(validation_test_folds)]
            # This means that for a training fold i, the training indices are training_indices[i], and the validation indices for a validation-training split j are val_test_indices[i][j]["val"] and val_test_indices[i][j]["test"]
            val_test_indices.append(split_val_test_indices_pairs)

            
            if chosen_training_fold == -1:
                training_ys = classifications[split_train_indices]
                val_test_ys = []
                for j in range(validation_test_folds):
                    val_test_ys.append([classifications[split_val_test_indices_pairs[j]["val"]], classifications[split_val_test_indices_pairs[j]["test"]]])

                training_hist = np.histogram(training_ys, bins = [1,2,3,4,5])[0]
                total_num_training = np.sum(training_hist)
                prop_by_class_train = training_hist / total_num_training

                train_val_scores = []
                for j in range(validation_test_folds):
                    val_ys = val_test_ys[j][0]
                    val_hist = np.histogram(val_ys, bins = [1,2,3,4,5])[0]
                    total_num_val = np.sum(val_hist)
                    prop_by_class_val = val_hist / total_num_val
                    val_train_match_score = np.sum(np.abs(prop_by_class_train - prop_by_class_val)**2)
                    print(f"Training fold {i}, validation fold {j} match score: {val_train_match_score}")
                    train_val_scores.append(val_train_match_score)
                best_val_fold = np.argmin(train_val_scores)
                # print(f"best val fold for training fold {i}: {best_val_fold}")
                best_val_folds.append(best_val_fold)
                best_val_fold_scores.append(train_val_scores[best_val_fold])


        if chosen_training_fold == -1:
            best_train_fold = np.argmin(best_val_fold_scores)
            corresponding_best_val_fold = best_val_folds[best_train_fold]
            chosen_training_fold = best_train_fold
            chosen_validation_fold = corresponding_best_val_fold
            print(f"Using best match with training fold {chosen_training_fold} and validation fold {chosen_validation_fold}")
        else:
            print(f"Using the specified training fold {chosen_training_fold} and validation fold {chosen_validation_fold}")
        
        selected_training_indices = training_indices[chosen_training_fold]
        # print(val_test_indices)
        selected_val_indices = val_test_indices[chosen_training_fold][chosen_validation_fold]["val"]
        selected_test_indices = val_test_indices[chosen_training_fold][chosen_validation_fold]["test"]

        
        # print(f"Training set final index: {samples[77].end_index}")
        # cutoff_index = samples[77].end_index
        
        train_data = torch.tensor(normalised_data[selected_training_indices], dtype=torch.float32).cpu()
        val_data = torch.tensor(normalised_data[selected_val_indices], dtype=torch.float32).cpu()
        test_data = torch.tensor(normalised_data[selected_test_indices], dtype=torch.float32).cpu()
        train_classes = torch.tensor(classifications[selected_training_indices] - 1, dtype=torch.long).cpu()
        val_classes = torch.tensor(classifications[selected_val_indices] - 1, dtype=torch.long).cpu()
        test_classes = torch.tensor(classifications[selected_test_indices] - 1, dtype=torch.long).cpu()
        
        print(f"Reducing to {prop_to_use * 100}% of loaded data")
        num_train = int(len(train_data)*prop_to_use)
        num_val = int(len(val_data)*prop_to_use)
        num_test = int(len(test_data)*prop_to_use)
        print(f"Keeping {num_train} for training")
        print(f"Keeping {num_val} for validation")
        print(f"Keeping {num_test} for testing")
        if randomise:
            keep_train = torch.randperm(len(train_data))[:num_train]
            keep_val = torch.randperm(len(val_data))[:num_val]
            keep_test = torch.randperm(len(test_data))[:num_test]
        else:
            print(num_train)
            keep_train = torch.arange(len(train_data))[:num_train]
            keep_val = torch.arange(len(val_data))[:num_val]
            keep_test = torch.arange(len(test_data))[:num_test]

        print(keep_train)
        train_data = train_data[keep_train]
        val_data = val_data[keep_val]
        test_data = test_data[keep_test]
        train_classes = train_classes[keep_train]
        val_classes = val_classes[keep_val]
        test_classes = test_classes[keep_test]
        if return_coords:
            train_coords = coords[selected_training_indices]
            val_coords = coords[selected_val_indices]
            test_coords = coords[selected_test_indices]
            train_sample_nums = np.array(sample_nums[selected_training_indices])-1
            val_sample_nums = np.array(sample_nums[selected_val_indices])-1
            test_sample_nums = np.array(sample_nums[selected_test_indices])-1
            print(np.unique(train_sample_nums))
            print(np.unique(val_sample_nums))
            print(np.unique(test_sample_nums))
            self.train_sample_nums = np.unique(train_sample_nums)
            self.val_sample_nums = np.unique(val_sample_nums)
            self.test_sample_nums = np.unique(test_sample_nums)

        train_dataset = TensorDataset(train_data, train_classes)
        val_dataset = TensorDataset(val_data, val_classes)
        test_dataset = TensorDataset(test_data, test_classes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=randomise, num_workers=14, pin_memory=True, prefetch_factor=20)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=randomise, num_workers=14, pin_memory=True, prefetch_factor=20)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=randomise, num_workers=14, pin_memory=True, prefetch_factor=20)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)

        if return_loaders:
            if return_coords:
                return train_loader, val_loader, test_loader, train_size, val_size, test_size, [train_coords, val_coords, test_coords], [train_sample_nums, val_sample_nums, test_sample_nums]
            else:
                return train_loader, val_loader, test_loader, train_size, val_size, test_size
        else:
            if return_coords:
                return train_data, train_classes, val_data, val_classes, test_data, test_classes, [train_coords, val_coords, test_coords], [train_sample_nums, val_sample_nums, test_sample_nums]
            else:
                return train_data, train_classes, val_data, val_classes, test_data, test_classes

    def initialise_logging(self, prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, extra_config=None):
        config = {
                "prop_to_use": prop_to_use,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "scheduler_step_size": scheduler_step_size,
                "scheduler_gamma": scheduler_gamma,
                "num_epochs": num_epochs,
                "entropy_weights": entropy_weights,
                "model_class": model_class.name,
                "batch_size": batch_size,
                "early_stop": early_stop,
                "training_fold_number": training_fold_number,
                "val_fold_number": val_fold_number,
                "number_of_training_folds": number_of_training_folds,
                "validation_test_folds": validation_test_folds,
                "dataset": "data.npy"
            }
        
        if extra_config is not None:
            config.update(extra_config)

        run = wandb.init(
            entity = "tjh200-university-of-cambridge",
            project = "TROPHY",
            config = config,
            tags = [model_class.name]
        )

        try:
            model_file = inspect.getfile(model_class)
            model_code_artifact = wandb.Artifact(
                name=f"{model_class.name}_code", 
                type="code"
            )

            model_code_artifact.add_file(model_file)
            run.log_artifact(model_code_artifact)
            
        except Exception as error:
            print(f"Warning: Could not save model source code: {error}")

        return run

    def train(self, prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size = 512, early_stop = True, optimise_for="loss", training_fold_number = -1, val_fold_number=1, number_of_training_folds = 3, validation_test_folds = 2, **kwargs):
        self.optimise_for = optimise_for
        print("Training Model")
        print(model_class.name)
        # run = self.initialise_logging(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds)
        if model_class == Core_CNN:
            train_loader, val_loader, test_loader, train_size, val_size, test_size = self.load_data(prop_to_use, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            # Fixed params
            apriori_relevance = kwargs.get("apriori_relevance", None).to(self.device)
            input_channels = kwargs['input_channels']

            # Hyperperams
            leak_rate = kwargs['leak_rate']
            dropout_rate = kwargs['dropout_rate']

            
            extra_config = {"apriori_relevance": apriori_relevance, "input_channels": input_channels, "leak_rate": leak_rate, "dropout_rate": dropout_rate}
            run = self.initialise_logging(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, extra_config=extra_config)
            
            model = Core_CNN(apriori_relevance, leak_rate, dropout_rate, input_channels).to(self.device)
            return self.propagate_model(run, model, train_loader, val_loader, test_loader, train_size, val_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights.to(self.device), early_stop) 
        
        if model_class == Core_CNN_TPL or model_class == Core_CNN_TPL2:
            train_loader, val_loader, test_loader, train_size, val_size, test_size = self.load_data(prop_to_use, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            # Fixed params
            apriori_relevance = kwargs.get("apriori_relevance", None).to(self.device)
            input_channels = kwargs['input_channels']

            # Hyperperams
            leak_rate = kwargs['leak_rate']
            dropout_rate = kwargs['dropout_rate']

            extra_config = {"apriori_relevance": apriori_relevance, "input_channels": input_channels, "leak_rate": leak_rate, "dropout_rate": dropout_rate}
            run = self.initialise_logging(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, extra_config=extra_config)

            model = model_class(apriori_relevance, leak_rate, dropout_rate, input_channels).to(self.device)
            return self.propagate_model(run, model, train_loader, val_loader, test_loader, train_size, val_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights.to(self.device), early_stop) 

        if model_class == SCNN_TPL or model_class == SCNN_TPL2 or model_class == SCNN_TPL3:
            train_loader, val_loader, test_loader, train_size, val_size, test_size = self.load_data(prop_to_use, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            # Fixed params
            apriori_relevance = kwargs.get("apriori_relevance", None).to(self.device)
            input_channels = kwargs['input_channels']

            # Hyperperams
            leak_rate = kwargs['leak_rate']
            dropout_rate = kwargs['dropout_rate']

            extra_config = {"apriori_relevance": apriori_relevance, "input_channels": input_channels, "leak_rate": leak_rate, "dropout_rate": dropout_rate}
            run = self.initialise_logging(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, extra_config=extra_config)

            model = model_class(apriori_relevance, leak_rate, dropout_rate, input_channels).to(self.device)
            return self.propagate_model(run, model, train_loader, val_loader, test_loader, train_size, val_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights.to(self.device), early_stop)

        if model_class == ViT:
            train_loader, val_loader, test_loader, train_size, val_size, test_size = self.load_data(prop_to_use, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            # Fixed params
            input_channels = kwargs['input_channels']
            input_length = kwargs['input_length']
            embed_kernel_size = kwargs['embed_kernel_size']
            embed_stride = kwargs['embed_stride']
            embedding_dims = kwargs['embedding_dims']

            # Hyperperams
            leak_rate = kwargs['leak_rate']
            dropout_rate = kwargs['dropout_rate']
            # run.config.update({"input_channels": input_channels, "input_length": input_length, "embed_kernel_size": embed_kernel_size, "embed_stride": embed_stride, "embedding_dims": embedding_dims, "leak_rate": leak_rate, "dropout_rate": dropout_rate})
            
            extra_config = {} # ADD this when finished
            run = self.initialise_logging(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size, early_stop, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, extra_config=extra_config)

            model = ViT(input_channels, input_length, embed_kernel_size, embed_stride, embedding_dims, leak_rate, dropout_rate).to(self.device)
            return self.propagate_model(run, model, train_loader, val_loader, test_loader, train_size, val_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights.to(self.device), early_stop)
        
        if model_class == XGB_Train:
            print("yes")
            # Here, we discount the validation data to give compariable results with other models (same training sets)
            train_data, train_classes, _, _, _, _ = self.load_data(prop_to_use, batch_size=batch_size, return_loaders=False, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            _, _, _, _, test_data, test_classes = self.load_data(1, batch_size=batch_size, return_loaders=False, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=training_fold_number, val_fold_number=val_fold_number)
            
            model = XGB_Train()
            return self.run_algo_model(model, train_data, train_classes, test_data, test_classes)
            # model.train(train_data, train_classes, test_data, test_classes)
            

    def run_algo_model(self, model, train_data, train_classes, test_data, test_classes):
        
        num_classes = 4
        labels = np.arange(num_classes)
        y_train = model.train(train_data, train_classes, test_data, test_classes)
        val_confusion_matrix = confusion_matrix(test_classes, y_train, labels=labels)
        val_sensitivity, val_specificity, val_accuracy, val_fpr, overall_accuracy = calculate_performance_stats_by_class(val_confusion_matrix)
        print(f"Accuracy: {val_accuracy}")
        print(f"Sensitivity: {val_sensitivity}")
        print(f"Specificity: {val_specificity}")
        print(f"FPR: {val_fpr}")
        print(f"Overall accuracy: {overall_accuracy}")
        return val_accuracy, val_sensitivity, val_specificity, val_fpr, overall_accuracy
        

    def propagate_model(self, run, model, train_loader, val_loader, test_loader, train_size, val_size, test_size, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, early_stop) -> float:
        criterion = nn.CrossEntropyLoss(weight=entropy_weights, label_smoothing=0.2)
        optimiser = optimise.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)
        take_final_score = False

        # best_val_sensitivity_score = 1000000
        best_val_stats = {}
        best_val_model_name = ""
        best_epoch = 0
        # best_val_sensitivity_test_sensitivty = -1
        epsilon = 1E-1
        offset = 0.2
        num_classes = 4
        labels = np.arange(num_classes)
        if self.optimise_for == "sensitivity_score": 
            best_val_score = 1000000
        if self.optimise_for == "loss":
            best_val_score = 1000000
        if self.optimise_for == "overall_accuracy":
            best_val_score = -1
    
        checkpoint_path = "/local/scratch/Data/TROPHY/Models/checkpoint.pth"
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            total_train_confusion_matrix = np.zeros((4,4))
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                num_X_samples = len(X_batch)
                X_batch = X_batch.view(num_X_samples, 1, 801)
                
                outputs = model(X_batch)
                # print(outputs)
        
                # Calculate training stats
                _, predicted = torch.max(outputs, 1)
                batch_c_mat = confusion_matrix(Y_batch.cpu(), predicted.cpu(), labels=labels)
                total_train_confusion_matrix += batch_c_mat
                train_sensitivity, _, _, _, _ = calculate_performance_stats_by_class(total_train_confusion_matrix)
                sensitivity_loss = 0
                for sensitivity in train_sensitivity:
                    if sensitivity > 0.3:
                        sensitivity_loss += 1/(sensitivity-offset+epsilon)
                    else:
                        sensitivity_loss += 10 + 1/(sensitivity + epsilon)

                loss = criterion(outputs, Y_batch) #+ sensitivity_loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                total_train_loss += loss.item() * X_batch.size(0)

            model.eval()
            total_eval_loss = 0.0
            total_eval_confusion_matrix = np.zeros((4,4))
            
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                    num_X_samples = len(X_batch)
                    X_batch = X_batch.view(num_X_samples, 1, 801)
        
                    
                    outputs = model(X_batch)

                    _, predicted = torch.max(outputs, 1)
                    batch_c_mat = confusion_matrix(Y_batch.cpu(), predicted.cpu(), labels=labels)
                    total_eval_confusion_matrix += batch_c_mat
                    # val_sensitivity, _, _, _ = calculate_performance_stats_by_class(total_eval_confusion_matrix)
                    # sensitivity_loss = 0
                    # for sensitivity in val_sensitivity:
                    #     if sensitivity > 0.3:
                    #         sensitivity_loss += 1/(sensitivity-offset+epsilon)
                    #     else:
                    #         sensitivity_loss += 10 + 1/(sensitivity + epsilon)

                    loss = criterion(outputs, Y_batch) #+ sensitivity_loss
                   
                    # Calculate training stats
                    total_eval_loss += loss.item() * X_batch.size(0)
                
            total_test_loss = 0.0
            total_test_confusion_matrix = np.zeros((4,4))
            
            with torch.no_grad():
                for X_batch, Y_batch in test_loader:
                    X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                    num_X_samples = len(X_batch)
                    X_batch = X_batch.view(num_X_samples, 1, 801)
        
                    
                    outputs = model(X_batch)

                    _, predicted = torch.max(outputs, 1)
                    batch_c_mat = confusion_matrix(Y_batch.cpu(), predicted.cpu(), labels=labels)
                    total_test_confusion_matrix += batch_c_mat

                    loss = criterion(outputs, Y_batch) #+ sensitivity_loss
                   
                    # Calculate training stats
                    total_test_loss += loss.item() * X_batch.size(0)
        
            scheduler.step()
            avg_train_loss = total_train_loss / train_size
            avg_val_loss = total_eval_loss / val_size
            avg_test_loss = total_test_loss / test_size

            epoch_train_sensitivity, epoch_train_specificity, epoch_train_accuracy, epoch_train_fpr, epoch_train_overall_accuracy = calculate_performance_stats_by_class(total_train_confusion_matrix)
            epoch_eval_sensitivity, epoch_eval_specificity, epoch_eval_accuracy, epoch_eval_fpr, epoch_eval_overall_accuracy = calculate_performance_stats_by_class(total_eval_confusion_matrix)
            epoch_test_sensitivity, epoch_test_specificity, epoch_test_accuracy, epoch_test_fpr, epoch_test_overall_accuracy = calculate_performance_stats_by_class(total_test_confusion_matrix)
            print(f"Epoch {epoch + 1}/{num_epochs} stats - learning rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"Loss: train {avg_train_loss}, val {avg_val_loss}, test {avg_test_loss}")
            print(f"Accuracy: train {epoch_train_accuracy}, val {epoch_eval_accuracy}, test {epoch_test_accuracy}")
            print(f"Sensitivity: train {epoch_train_sensitivity}, val {epoch_eval_sensitivity}, test {epoch_test_sensitivity}")
            print(f"Specificity: train {epoch_train_specificity}, val {epoch_eval_specificity}, test {epoch_test_specificity}")
            print(f"FPR: train {epoch_train_fpr}, val {epoch_eval_fpr}, test {epoch_test_fpr}")
            print(f"Overall accuracy: train {epoch_train_overall_accuracy}, val {epoch_eval_overall_accuracy}, test {epoch_test_overall_accuracy}")

            val_sensitivity_score = 0
            test_sensitivity_score = 0
            epsilon =  1E-5
            for sensitivity in epoch_eval_sensitivity:
                val_sensitivity_score += 1/(sensitivity+epsilon)
            for sensitivity in epoch_test_sensitivity:
                test_sensitivity_score += 1/(sensitivity+epsilon)

            if self.optimise_for == "sensitivity_score": 
                val_score = val_sensitivity_score
                test_score = test_sensitivity_score
                condition = val_score < best_val_score
            if self.optimise_for == "loss":
                val_score = avg_val_loss
                test_score = avg_test_loss
                condition = val_score < best_val_score
            if self.optimise_for == "overall_accuracy":
                val_score = epoch_eval_overall_accuracy
                test_score = epoch_test_overall_accuracy
                condition = val_score > best_val_score

            if condition or take_final_score == True:
                best_val_score = val_score
                best_val_test_score = test_score
                best_val_stats = {'train_sensitivity': epoch_train_sensitivity, 'train_specificity': epoch_train_specificity, 'train_accuracy': epoch_train_accuracy, 'train_fpr': epoch_train_fpr,
                                'val_sensitivity': epoch_eval_sensitivity, 'val_specificity': epoch_eval_specificity, 'val_accuracy': epoch_eval_accuracy, 'val_fpr': epoch_eval_fpr}
                best_metrics = {"epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "test_loss": avg_test_loss,
                    "train_accuracy_0": epoch_train_accuracy[0],
                    "train_accuracy_1": epoch_train_accuracy[1],
                    "train_accuracy_2": epoch_train_accuracy[2],
                    "train_accuracy_3": epoch_train_accuracy[3],
                    "val_accuracy_0": epoch_eval_accuracy[0],
                    "val_accuracy_1": epoch_eval_accuracy[1],
                    "val_accuracy_2": epoch_eval_accuracy[2],
                    "val_accuracy_3": epoch_eval_accuracy[3],
                    "test_accuracy_0": epoch_test_accuracy[0],
                    "test_accuracy_1": epoch_test_accuracy[1],
                    "test_accuracy_2": epoch_test_accuracy[2],
                    "test_accuracy_3": epoch_test_accuracy[3],
                    "train_sensitivity_0": epoch_train_sensitivity[0],
                    "train_sensitivity_1": epoch_train_sensitivity[1],
                    "train_sensitivity_2": epoch_train_sensitivity[2],
                    "train_sensitivity_3": epoch_train_sensitivity[3],
                    "val_sensitivity_0": epoch_eval_sensitivity[0],
                    "val_sensitivity_1": epoch_eval_sensitivity[1],
                    "val_sensitivity_2": epoch_eval_sensitivity[2],
                    "val_sensitivity_3": epoch_eval_sensitivity[3],
                    "test_sensitivity_0": epoch_test_sensitivity[0],
                    "test_sensitivity_1": epoch_test_sensitivity[1],
                    "test_sensitivity_2": epoch_test_sensitivity[2],
                    "test_sensitivity_3": epoch_test_sensitivity[3],
                    "train_specificity_0": epoch_train_specificity[0],
                    "train_specificity_1": epoch_train_specificity[1],
                    "train_specificity_2": epoch_train_specificity[2],
                    "train_specificity_3": epoch_train_specificity[3],
                    "val_specificity_0": epoch_eval_specificity[0],
                    "val_specificity_1": epoch_eval_specificity[1],
                    "val_specificity_2": epoch_eval_specificity[2],
                    "val_specificity_3": epoch_eval_specificity[3],
                    "test_specificity_0": epoch_test_specificity[0],
                    "test_specificity_1": epoch_test_specificity[1],
                    "test_specificity_2": epoch_test_specificity[2],
                    "test_specificity_3": epoch_test_specificity[3],
                    "train_fpr_0": epoch_train_fpr[0],
                    "train_fpr_1": epoch_train_fpr[1],
                    "train_fpr_2": epoch_train_fpr[2],
                    "train_fpr_3": epoch_train_fpr[3],
                    "val_fpr_0": epoch_eval_fpr[0],
                    "val_fpr_1": epoch_eval_fpr[1],
                    "val_fpr_2": epoch_eval_fpr[2],
                    "val_fpr_3": epoch_eval_fpr[3],
                    "test_fpr_0": epoch_test_fpr[0],
                    "test_fpr_1": epoch_test_fpr[1],
                    "test_fpr_2": epoch_test_fpr[2],
                    "test_fpr_3": epoch_test_fpr[3],
                    "train_overall_accuracy": epoch_train_overall_accuracy,
                    "val_overall_accuracy": epoch_eval_overall_accuracy,
                    "test_overall_accuracy": epoch_test_overall_accuracy,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "val_sensitivity_score": val_sensitivity_score,
                    "test_sensitivity_score": test_sensitivity_score}
                print(f"!!!New max val {self.optimise_for} score {val_score}, corresponding test sensitivity score {test_score}")
                val_score_string = f"{val_score:.6f}"
        
                torch.save({
                    "model": model.name,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                    "train_stats": [epoch_train_sensitivity, epoch_train_specificity, epoch_train_accuracy, epoch_train_fpr],
                    "val_stats": [epoch_eval_sensitivity, epoch_eval_specificity, epoch_eval_accuracy, epoch_eval_fpr],
                    "test_stats": [epoch_test_sensitivity, epoch_test_specificity, epoch_test_accuracy, epoch_test_fpr]},
                    self.models_path + val_score_string + model.name+ run.name + ".pth")
                
                if best_val_model_name != "":
                    old_best_model_path = self.models_path + best_val_model_name
                    
                    if os.path.exists(old_best_model_path):
                        os.remove(old_best_model_path)
                        print("Removed previous best model")
                    else:
                        print("Previous best file not found")
                        
                best_val_model_name = val_score_string + model.name+ run.name+".pth"
            
            # if self.optimise_for == "loss":
            #     if avg_val_loss < best_val_score:
            #         best_val_score = avg_val_loss
            #         torch.save(model.state_dict(), checkpoint_path)
            # if self.optimise_for == "sensitivity_score":
            #     if val_sensitivity_score < best_val_score:
            #         best_val_score = val_sensitivity_score
            #         torch.save(model.state_dict(), checkpoint_path)
            
            
            run.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "test_loss": avg_test_loss,
                "train_accuracy_0": epoch_train_accuracy[0],
                "train_accuracy_1": epoch_train_accuracy[1],
                "train_accuracy_2": epoch_train_accuracy[2],
                "train_accuracy_3": epoch_train_accuracy[3],
                "val_accuracy_0": epoch_eval_accuracy[0],
                "val_accuracy_1": epoch_eval_accuracy[1],
                "val_accuracy_2": epoch_eval_accuracy[2],
                "val_accuracy_3": epoch_eval_accuracy[3],
                "test_accuracy_0": epoch_test_accuracy[0],
                "test_accuracy_1": epoch_test_accuracy[1],
                "test_accuracy_2": epoch_test_accuracy[2],
                "test_accuracy_3": epoch_test_accuracy[3],
                "train_sensitivity_0": epoch_train_sensitivity[0],
                "train_sensitivity_1": epoch_train_sensitivity[1],
                "train_sensitivity_2": epoch_train_sensitivity[2],
                "train_sensitivity_3": epoch_train_sensitivity[3],
                "val_sensitivity_0": epoch_eval_sensitivity[0],
                "val_sensitivity_1": epoch_eval_sensitivity[1],
                "val_sensitivity_2": epoch_eval_sensitivity[2],
                "val_sensitivity_3": epoch_eval_sensitivity[3],
                "test_sensitivity_0": epoch_test_sensitivity[0],
                "test_sensitivity_1": epoch_test_sensitivity[1],
                "test_sensitivity_2": epoch_test_sensitivity[2],
                "test_sensitivity_3": epoch_test_sensitivity[3],
                "train_specificity_0": epoch_train_specificity[0],
                "train_specificity_1": epoch_train_specificity[1],
                "train_specificity_2": epoch_train_specificity[2],
                "train_specificity_3": epoch_train_specificity[3],
                "val_specificity_0": epoch_eval_specificity[0],
                "val_specificity_1": epoch_eval_specificity[1],
                "val_specificity_2": epoch_eval_specificity[2],
                "val_specificity_3": epoch_eval_specificity[3],
                "test_specificity_0": epoch_test_specificity[0],
                "test_specificity_1": epoch_test_specificity[1],
                "test_specificity_2": epoch_test_specificity[2],
                "test_specificity_3": epoch_test_specificity[3],
                "train_fpr_0": epoch_train_fpr[0],
                "train_fpr_1": epoch_train_fpr[1],
                "train_fpr_2": epoch_train_fpr[2],
                "train_fpr_3": epoch_train_fpr[3],
                "val_fpr_0": epoch_eval_fpr[0],
                "val_fpr_1": epoch_eval_fpr[1],
                "val_fpr_2": epoch_eval_fpr[2],
                "val_fpr_3": epoch_eval_fpr[3],
                "test_fpr_0": epoch_test_fpr[0],
                "test_fpr_1": epoch_test_fpr[1],
                "test_fpr_2": epoch_test_fpr[2],
                "test_fpr_3": epoch_test_fpr[3],
                "train_overall_accuracy": epoch_train_overall_accuracy,
                "val_overall_accuracy": epoch_eval_overall_accuracy,
                "test_overall_accuracy": epoch_test_overall_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
                "val_sensitivity_score": val_sensitivity_score,
                "test_sensitivity_score": test_sensitivity_score})
            print("Params: ",sum(p.numel() for p in model.parameters()))
            
            
            if early_stop and epoch > 30:
                if np.isinf(avg_val_loss) or np.isnan(avg_val_loss) or val_sensitivity_score > 5000 or avg_val_loss > 2 * avg_train_loss:
                    print(f"Skipping the rest of training, best: {best_val_score}")

                    best_model_artifact = wandb.Artifact('best-model', type='model')
                    best_model_artifact.add_file(self.models_path + best_val_model_name)
                    run.log_artifact(best_model_artifact)

                    val_score_string = f"{val_score:.6f}"
                    final_val_model_name = val_score_string + model.name+ run.name+".pth"
                    torch.save({
                        "model": model.name,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                        "train_stats": [epoch_train_sensitivity, epoch_train_specificity, epoch_train_accuracy, epoch_train_fpr],
                        "val_stats": [epoch_eval_sensitivity, epoch_eval_specificity, epoch_eval_accuracy, epoch_eval_fpr],
                        "test_stats": [epoch_test_sensitivity, epoch_test_specificity, epoch_test_accuracy, epoch_test_fpr]},
                        self.models_path + final_val_model_name)
                    final_model_artifact = wandb.Artifact('final-model', type='model')
                    final_model_artifact.add_file(self.models_path + final_val_model_name)
                    run.log_artifact(final_model_artifact)
                    for metric_name, metric_value in best_metrics.items():
                        run.summary[f"best_{metric_name}"] = metric_value

                    run.finish()

                    if best_val_model_name != "":
                        old_best_model_path = self.models_path + best_val_model_name
                        
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)
                            print("Removed best model")
                        else:
                            print("Previous best file not found")
                    
                    final_model_path = self.models_path + final_val_model_name
                    if os.path.exists(final_model_path):
                        os.remove(final_model_path)
                        print("Removed final model")
                    return best_val_score, best_val_stats
                    break
                
        print(f"Training complete, best val sensitivity: {best_val_score} with corresponding test sensitivity: {best_val_test_score}")

        best_model_artifact = wandb.Artifact('best-model', type='model')
        best_model_artifact.add_file(self.models_path + best_val_model_name)
        run.log_artifact(best_model_artifact)

        val_score_string = f"{val_score:.6f}"
        final_val_model_name = val_score_string + model.name+ run.name+".pth"
        torch.save({
            "model": model.name,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
            "train_stats": [epoch_train_sensitivity, epoch_train_specificity, epoch_train_accuracy, epoch_train_fpr],
            "val_stats": [epoch_eval_sensitivity, epoch_eval_specificity, epoch_eval_accuracy, epoch_eval_fpr],
            "test_stats": [epoch_test_sensitivity, epoch_test_specificity, epoch_test_accuracy, epoch_test_fpr]},
            self.models_path + final_val_model_name)
        final_model_artifact = wandb.Artifact('final-model', type='model')
        final_model_artifact.add_file(self.models_path + final_val_model_name)
        run.log_artifact(final_model_artifact)
        for metric_name, metric_value in best_metrics.items():
            run.summary[f"best_{metric_name}"] = metric_value

        run.finish()

        if best_val_model_name != "":
            old_best_model_path = self.models_path + best_val_model_name
            
            if os.path.exists(old_best_model_path):
                os.remove(old_best_model_path)
                print("Removed best model")
            else:
                print("Previous best file not found")
        
        final_model_path = self.models_path + final_val_model_name
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
            print("Removed final model")
        return best_val_score, best_val_stats # Add a way to start returning the best specificity and accuracy per model
        
        # Best:  75.87% 
    
    
    def evaluate_test_data(self, model, model_path, prop_to_use, num_classes, batch_size, eval_on="test", training_fold_number = -1, val_fold_number=1, number_of_training_folds = 4, validation_test_folds = 2):
        labels = np.arange(num_classes)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.eval()

        train_loader, val_loader, test_loader, _, _, _, coords, sample_nums = self.load_data(prop_to_use, batch_size, number_of_training_folds, validation_test_folds, training_fold_number, val_fold_number, randomise=False, return_coords=True)
     
        if eval_on == "train":
            data_loader = train_loader
            coord_index = 0
        elif eval_on == "val":
            data_loader = val_loader
            coord_index = 1
        elif eval_on == "test":
            data_loader = test_loader
            coord_index = 2
        

        total_eval_confusion_matrix = np.zeros((4,4))
        predictions = []
        true_outputs = []
        raw_inputs = np.empty((0, 801))
        
        with torch.no_grad():
            print("Params: ",sum(p.numel() for p in model.parameters()))
            for X_batch, Y_batch in data_loader:
                X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                num_X_samples = len(X_batch)
                X_batch = X_batch.view(num_X_samples, 1, 801)
                
                # Model Prediction
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                
                predictions = np.concat([predictions, predicted.cpu()])
                true_outputs = np.concat([true_outputs, Y_batch.cpu()])
                raw_inputs = np.concat([raw_inputs, (X_batch.squeeze(1)).cpu()]) 

                batch_c_mat = confusion_matrix(Y_batch.cpu(), predicted.cpu(), labels=labels)
                total_eval_confusion_matrix += batch_c_mat
        
        return total_eval_confusion_matrix, predictions, true_outputs, raw_inputs, coords[coord_index], sample_nums[coord_index]

    def evaluate_multiple_models(self, model_class, models_params, prop_to_use, num_classes, batch_size, apriori_relevance, eval_on="test", training_fold_number = -1, val_fold_number=1, number_of_training_folds = 4, validation_test_folds = 2):
        from scipy import stats
        labels = np.arange(num_classes)
        
        # Returning coords also allows the use of self.train_sample_nums etc.
        train_loader, val_loader, test_loader, _, _, _, coords, sample_nums = self.load_data(prop_to_use, batch_size, number_of_training_folds, validation_test_folds, training_fold_number, val_fold_number, randomise=False, return_coords=True)
     
        if eval_on == "train":
            data_loader = train_loader
            coord_index = 0
        elif eval_on == "val":
            data_loader = val_loader
            coord_index = 1
        elif eval_on == "test":
            data_loader = test_loader
            coord_index = 2

        model_predictions = []
        model_confusion_matrices = []
        
        for model_params in models_params:
            print(model_params)
            evaluation_model = model_class(apriori_relevance, model_params['config']['leak_rate'], model_params['config']['dropout_rate'], 1)
            evaluation_model = evaluation_model.to(self.device)
            evaluation_model.load_state_dict(torch.load(model_params['model_path'])['model_state_dict'])
            evaluation_model.eval()
            
            total_eval_confusion_matrix = np.zeros((4,4))
            predictions = []
            true_outputs = []
            raw_inputs = np.empty((0, 801))
            
            with torch.no_grad():
                print("Params: ",sum(p.numel() for p in evaluation_model.parameters()))
                for X_batch, Y_batch in data_loader:
                    X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                    num_X_samples = len(X_batch)
                    X_batch = X_batch.view(num_X_samples, 1, 801)
                    
                    # Model Prediction
                    outputs = evaluation_model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    
                    predictions = np.concat([predictions, predicted.cpu()])
                    true_outputs = np.concat([true_outputs, Y_batch.cpu()])
                    raw_inputs = np.concat([raw_inputs, (X_batch.squeeze(1)).cpu()]) 

                    batch_c_mat = confusion_matrix(Y_batch.cpu(), predicted.cpu(), labels=labels)
                    total_eval_confusion_matrix += batch_c_mat
            
                # total_eval_confusion_matrix, predictions, true_outputs, raw_inputs, coords[coord_index], sample_nums[coord_index]
            model_predictions.append(predictions)
            model_confusion_matrices.append(total_eval_confusion_matrix)
        final_predictions = stats.mode(model_predictions, axis=0)[0]
        # final_predictions = np.mode(model_predictions, axis=0)
        final_confusion_matrix = np.sum(model_confusion_matrices, axis=0)
        return final_confusion_matrix, final_predictions, true_outputs, raw_inputs, coords[coord_index], sample_nums[coord_index]
    
    def chat_evaluate_multiple_models(self, model_class, models_params, prop_to_use, num_classes, batch_size, 
                            apriori_relevance, eval_on="test", training_fold_number=-1, 
                            val_fold_number=1, number_of_training_folds=4, validation_test_folds=2):
        """
        Evaluate multiple models on a dataset and combine predictions using voting.
        Optimized to process the dataset only once for all models.
        """
        import torch
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from collections import Counter

        labels = np.arange(num_classes)

        # Load data once
        train_loader, val_loader, test_loader, _, _, _, coords, sample_nums = self.load_data(
            prop_to_use, batch_size, number_of_training_folds, validation_test_folds, 
            training_fold_number, val_fold_number, randomise=False, return_coords=True
        )

        if eval_on == "train":
            data_loader = train_loader
            coord_index = 0
            evaluated_sample_nums = self.train_sample_nums
        elif eval_on == "val":
            data_loader = val_loader
            coord_index = 1
            evaluated_sample_nums = self.val_sample_nums
        elif eval_on == "test":
            data_loader = test_loader
            coord_index = 2
            evaluated_sample_nums = self.test_sample_nums

        # Preload all models to memory
        print(f"Preloading {len(models_params)} models...")
        models = []
        for model_params in models_params:
            model = model_class(apriori_relevance, 
                                model_params['config']['leak_rate'], 
                                model_params['config']['dropout_rate'], 1)
            model = model.to(self.device)
            model.load_state_dict(torch.load(model_params['model_path'])['model_state_dict'])
            model.eval()
            models.append(model)
            print(f"Loaded model: {model_params['model_path']}")

        # Process dataset once for all models
        print("Processing dataset...")
        all_predictions = []
        true_outputs = None
        raw_inputs = None

        with torch.no_grad():
            batch_count = 0
            for X_batch, Y_batch in data_loader:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processing batch {batch_count}...")
                    
                X_batch, Y_batch = X_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                num_X_samples = len(X_batch)
                X_batch = X_batch.view(num_X_samples, 1, 801)
                
                # Save true outputs and raw inputs once
                if true_outputs is None:
                    true_outputs = Y_batch.cpu().numpy()
                    raw_inputs = X_batch.squeeze(1).cpu().numpy()
                else:
                    true_outputs = np.concatenate([true_outputs, Y_batch.cpu().numpy()])
                    raw_inputs = np.concatenate([raw_inputs, X_batch.squeeze(1).cpu().numpy()])
                
                # Get predictions from all models for this batch
                batch_predictions = []
                for model in models:
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    batch_predictions.append(predicted.cpu().numpy())
                
                # Initialize or extend predictions arrays
                if not all_predictions:
                    all_predictions = [pred for pred in batch_predictions]
                else:
                    for i, pred in enumerate(batch_predictions):
                        all_predictions[i] = np.concatenate([all_predictions[i], pred])

        # Convert to numpy arrays for easier handling
        all_predictions = np.array(all_predictions)

        # Create ensemble predictions using majority voting
        ensemble_predictions = np.zeros_like(true_outputs)
        for i in range(len(true_outputs)):
            votes = all_predictions[:, i]  # Get all model predictions for this sample
            # Find most common prediction (mode)
            most_common = Counter(votes).most_common(1)
            ensemble_predictions[i] = most_common[0][0]

        # Calculate final confusion matrix
        final_confusion_matrix = confusion_matrix(true_outputs, ensemble_predictions, labels=labels)

        return final_confusion_matrix, ensemble_predictions, true_outputs, raw_inputs, coords[coord_index], sample_nums[coord_index], evaluated_sample_nums