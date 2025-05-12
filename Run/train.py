import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
crime_dir = os.path.abspath(os.path.join(script_path, ".."))
sys.path.insert(0, crime_dir)

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
from Models.XGBoost import XGB_Train
from Models.Core_CNN_TPL2 import Core_CNN_TPL2
from Models.SCNN_TPL3 import SCNN_TPL3
from Models.SCNN_TPL2 import SCNN_TPL2
from Models.ViT import ViT
from Processing.Trainer import Trainer
from hyperopt import hp, fmin, atpe, space_eval, STATUS_OK, Trials
from functools import partial
import warnings
np.warnings = warnings
warnings.filterwarnings("ignore")
import wandb

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_config", type=int, required=True, help="Dataset configuration number")
parser.add_argument("--prop_to_use", type=float, required=True, help="Proportion of data to use")
parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
parser.add_argument("--model_class", type=str, required=True, help="Model class to use")
parser.add_argument("--input_channels", type=int, required=True, help="Number of input channels")
parser.add_argument("--number_of_training_folds", type=int, required=True, help="Number of training folds")
parser.add_argument("--validation_test_folds", type=int, required=True, help="Number of validation test folds")
parser.add_argument("--leak_rate", type=float, required=True, help="Leak rate")
parser.add_argument("--dropout_rate", type=float, required=True, help="Dropout rate")
parser.add_argument("--entropy_weight_0", type=float, required=True, help="Entropy weight 0")
parser.add_argument("--entropy_weight_1", type=float, required=True, help="Entropy weight 1")
parser.add_argument("--entropy_weight_2", type=float, required=True, help="Entropy weight 2")
parser.add_argument("--entropy_weight_3", type=float, required=True, help="Entropy weight 3")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay")
parser.add_argument("--scheduler_step_size", type=int, required=True, help="Scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, required=True, help="Scheduler gamma")
parser.add_argument("--early_stop", type=str, required=True, help="Early stop")
parser.add_argument("--apriori_relevance", type=str, required=True, help="Apriori relevance - use default for all 1s")
parser.add_argument("--optimise_for", type=str, required=True, help="Optimise for either 'loss', 'sensitivity_score', or 'overall_accuracy")


run_args = parser.parse_args()
print(run_args)
print(f"Running process {run_args.dataset_config}")
# Define fixed parameters
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

bool_dict = {
    "true": True,
    "false": False
}
dataset_config = run_args.dataset_config
prop_to_use = run_args.prop_to_use
num_epochs = run_args.num_epochs
batch_size = run_args.batch_size
model_class_name = run_args.model_class
model_class = model_class_dict[model_class_name]

input_channels = run_args.input_channels
number_of_training_folds = run_args.number_of_training_folds
validation_test_folds = run_args.validation_test_folds
if run_args.apriori_relevance == "default":
    apriori_relevance = torch.ones(801)
else:
    print("Implement apriori relevance customisation")

train_fold_number = dataset_config // validation_test_folds
val_fold_number = dataset_config % validation_test_folds

learning_rate = run_args.learning_rate
weight_decay = run_args.weight_decay
scheduler_step_size = run_args.scheduler_step_size
scheduler_gamma = run_args.scheduler_gamma
leak_rate = run_args.leak_rate
dropout_rate = run_args.dropout_rate
entropy_weights = torch.tensor([run_args.entropy_weight_0,run_args.entropy_weight_1,run_args.entropy_weight_2,run_args.entropy_weight_3]).float()
early_stop_string = run_args.early_stop
early_stop = bool_dict[early_stop_string]
optimise_for = run_args.optimise_for


trainer = Trainer()
train_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=train_fold_number, val_fold_number = val_fold_number, early_stop=early_stop, optimise_for=optimise_for,
            apriori_relevance = apriori_relevance, input_channels = input_channels, leak_rate = leak_rate, dropout_rate = dropout_rate)

# return {"loss": train_score[0], "status": STATUS_OK}