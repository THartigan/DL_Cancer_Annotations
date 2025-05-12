import sys
import os
script_path = os.getcwd()
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
from Processing.Trainer import Trainer
from hyperopt import hp, fmin, atpe, space_eval, STATUS_OK, Trials
from functools import partial
import warnings
np.warnings = warnings
warnings.filterwarnings("ignore")

import argparse

process_id = 0

print(f"Running process {process_id}")

# Define fixed parameters
prop_to_use = 1
num_epochs = 100
model_class = XGB_Train
apriori_relevance = torch.ones(801)
input_channels = 1


learning_rate = 5E-4
weight_decay = 0.1
scheduler_step_size = 10 #args['scheduler_step_size']
scheduler_gamma = 0.5 #args['scheduler_gamma']
leak_rate = 0.1
dropout_rate = 0.7
entropy_weights = torch.tensor([1,0.7137419678717736,1,1])

num_folds = 10
accuracies = []
sensitivities = []
specificities = []
fprs = []
overall_accuracies = []

for fold in range(num_folds):
    trainer = Trainer()
    val_accuracy, val_sensitivity, val_specificity, val_fpr, accuracy_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size=32400, early_stop=False, fold_number=fold)
    accuracies.append(val_accuracy)
    sensitivities.append(val_sensitivity)
    specificities.append(val_specificity)
    fprs.append(val_fpr)
    overall_accuracies.append(accuracy_score)


print(f"Mean overall accuracy: {np.mean(overall_accuracies)}")
print(f"Standard deviation overall accuracy: {np.std(overall_accuracies)}")
print(f"Mean sensitivity: {np.mean(sensitivities, axis=0)}")
print(f"Standard deviation sensitivity: {np.std(sensitivities, axis=0)}")
print(f"Mean specificity: {np.mean(specificities, axis=0)}")
print(f"Standard deviation specificity: {np.std(specificities, axis=0)}")

# train_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size=32400, early_stop=False, fold_number=5,
#             apriori_relevance = apriori_relevance, input_channels = input_channels, leak_rate = leak_rate, dropout_rate = dropout_rate)
# print(train_score)




# # define an objective function
# def objective(args):
#     learning_rate = args['learning_rate']
#     weight_decay = args['weight_decay']
#     scheduler_step_size = 10 #args['scheduler_step_size']
#     scheduler_gamma = 0.5 #args['scheduler_gamma']
#     leak_rate = args['leak_rate']
#     dropout_rate = args['dropout_rate']
#     entropy_weights = torch.tensor([args['entropy_weight_0'],args['entropy_weight_1'],args['entropy_weight_2'],args['entropy_weight_3']]).float()

#     trainer = Trainer()
#     train_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size=3600, fold_number=run_args.process_id,
#               apriori_relevance = apriori_relevance, input_channels = input_channels, leak_rate = leak_rate, dropout_rate = dropout_rate)
#     print(train_score)
#     return {"loss": train_score[0], "status": STATUS_OK}

# # define a search space
# space = {
#     "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
#     "weight_decay": hp.loguniform("weight_decay", np.log(1E-5), 0.2),
#     # "scheduler_step_size": hp.quniform("scheduler_step_size", 1, 50, 1),
#     # "scheduler_gamma": hp.uniform("scheduler_gamma", 0, 1),
#     "leak_rate": hp.loguniform("leak_rate", np.log(1E-2), 0.3),
#     "dropout_rate": hp.uniform("dropout_rate", 0.25, 0.9),
#     "entropy_weight_0": hp.uniform("entropy_weight_0", 0.1, 1),
#     "entropy_weight_1": hp.uniform("entropy_weight_1", 0.1, 1),
#     "entropy_weight_2": hp.uniform("entropy_weight_2", 0.1, 1),
#     "entropy_weight_3": hp.uniform("entropy_weight_3", 0.1, 1)
# }

# trials = Trials()

# algorithm = atpe.suggest

# # minimize the objective over the space
# best = fmin(objective, space, algo=algorithm, max_evals=100, trials=trials)

# print(best)
# print(space_eval(space, best))
# save_directory = "/local/scratch/Data/TROPHY/Models/"
# np.save(save_directory + "best_" + model_class.name + str(process_id), best )
# np.save(save_directory + "trials_" + model_class.name + str(process_id), trials)