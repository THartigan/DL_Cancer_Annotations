import os
import sys
# resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1) add CRIME folder so Processing/Models can be found
crime_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, crime_root)

# # 2) add external crime package (adjust if your structure differs)
# crime_pkg = os.path.abspath(os.path.join(crime_root, '..', 'CRIME_Package', 'CRIME'))
# sys.path.insert(0, crime_pkg)

import lime.explanation
import torch
import numpy as np
from Processing.Sample import Sample
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
import torch.optim as optimise
from torch.optim.lr_scheduler import StepLR
from Models.Core_CNN import Core_CNN
from Models.Core_CNN_TPL import Core_CNN_TPL
from Models.SCNN_TPL import SCNN_TPL
from Models.SCNN_TPL3 import SCNN_TPL3
from Processing.Trainer import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb

# data = Trainer.load_data(1, )

import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import crime as cr
from crime.CRIME_functions import run_CRIME
import crime.lime_processing_functions as lpf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from Processing.model_handling import get_fold_numbers
from Processing.model_handling import get_models_from_sweep
from Models.CombinationModel import CombinationModel

# First we need to import the data

sweep_id = "wh844aab"
entity = "tjh200-university-of-cambridge"
project = "TROPHY"
threshold = 0.55
threshold_quantity = "val_overall_accuracy"
eval_on = "test"
num_classes = 4
num_to_use_per_class = 1000
prop_to_load = 1
assume_model_classifications = True

api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
sweep_config = sweep.config

dataset_config = sweep_config['parameters']['dataset_config']['value']
num_training_folds = sweep_config['parameters']['number_of_training_folds']['value']
num_val_test_folds = sweep_config['parameters']['validation_test_folds']['value']
prop_to_use = sweep_config['parameters']['prop_to_use']['value']
trained_batch_size = sweep_config['parameters']['batch_size']['value']
model_class = sweep_config['parameters']['model_class']['values'][0]

train_fold_number, val_fold_number = get_fold_numbers(dataset_config, num_val_test_folds)


trainer = Trainer()
train_data, train_classes, val_data, val_classes, test_data, test_classes, [train_coords, val_coords, test_coords], [train_sample_nums, val_sample_nums, test_sample_nums]  = trainer.load_data(prop_to_load, trained_batch_size, num_training_folds, num_val_test_folds, train_fold_number, val_fold_number, return_loaders=False, return_coords=True, randomise=False)
print(train_classes)
selected_models = get_models_from_sweep(sweep_id, threshold_quantity, threshold)
combination_model = CombinationModel(model_class, selected_models, num_classes, combination_mode="vote_probs")
if assume_model_classifications:
    train_classes = torch.tensor(np.argmax(combination_model.predict(train_data), axis=1))
    val_classes = torch.tensor(np.argmax(combination_model.predict(val_data), axis=1))
    test_classes = torch.tensor(np.argmax(combination_model.predict(test_data), axis=1))

final_train_data = []
final_train_classes = []
final_val_data = []
final_val_classes = []
final_test_data = []
final_test_classes = []

for identity in range(num_classes):
    train_identity_indices = np.where(train_classes == identity)[0]
    val_identity_indices = np.where(val_classes == identity)[0]
    test_identity_indices = np.where(test_classes == identity)[0]
    if len(train_identity_indices) > num_to_use_per_class:
        train_identity_indices = np.random.choice(train_identity_indices, num_to_use_per_class, replace=False)
    if len(val_identity_indices) > num_to_use_per_class:
        val_identity_indices = np.random.choice(val_identity_indices, num_to_use_per_class, replace=False)
    if len(test_identity_indices) > num_to_use_per_class:
        test_identity_indices = np.random.choice(test_identity_indices, num_to_use_per_class, replace=False)
    identity_train_data = train_data[train_identity_indices][:num_to_use_per_class]
    identity_train_classes = train_classes[train_identity_indices][:num_to_use_per_class]
    identity_val_data = val_data[val_identity_indices][:num_to_use_per_class]
    identity_val_classes = val_classes[val_identity_indices][:num_to_use_per_class]
    identity_test_data = test_data[test_identity_indices][:num_to_use_per_class]
    identity_test_classes = test_classes[test_identity_indices][:num_to_use_per_class]
    final_train_data = np.concatenate((final_train_data, identity_train_data), axis=0) if len(final_train_data) > 0 else identity_train_data
    final_train_classes = np.concatenate((final_train_classes, identity_train_classes), axis=0) if len(final_train_classes) > 0 else identity_train_classes
    final_val_data = np.concatenate((final_val_data, identity_val_data), axis=0) if len(final_val_data) > 0 else identity_val_data
    final_val_classes = np.concatenate((final_val_classes, identity_val_classes), axis=0) if len(final_val_classes) > 0 else identity_val_classes
    final_test_data = np.concatenate((final_test_data, identity_test_data), axis=0) if len(final_test_data) > 0 else identity_test_data 
    final_test_classes = np.concatenate((final_test_classes, identity_test_classes), axis=0) if len(final_test_classes) > 0 else identity_test_classes

if eval_on == "train":
    run_data = final_train_data
    run_classes = final_train_classes
elif eval_on == "val":
    run_data = final_val_data
    run_classes = final_val_classes
elif eval_on == "test":
    run_data = final_test_data
    run_classes = final_test_classes


explainer = lpf.spectra_explainer(run_data, len(run_data[0]))
categories = [run_data[run_classes == i] for i in range(4)]
wavenumbers = np.load('/local/scratch/Data/TROPHY/numpy/wavenumbers.npy')

lime_data, category_indicator, spectra_indicator, mean_spectra_list = lpf.calculate_lime(categories, explainer, wavenumbers, combination_model.predict)

run_identifier = f"CRIME_{sweep_id}_{eval_on}_{num_to_use_per_class}_{threshold}_{assume_model_classifications}"
save_dir = f"/local/scratch/code/TROPHY/colon_data_analysis/CRIME/Results/{run_identifier}/"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'lime_data.npy'), lime_data)
np.save(os.path.join(save_dir, 'category_indicator.npy'), category_indicator)
np.save(os.path.join(save_dir, 'spectra_indicator.npy'), spectra_indicator)
np.save(os.path.join(save_dir, 'mean_spectra_list.npy'), mean_spectra_list)
