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
from Processing.Trainer import Trainer
from hyperopt import hp, fmin, atpe, space_eval, STATUS_OK, Trials
from functools import partial
import warnings
np.warnings = warnings
warnings.filterwarnings("ignore")

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--process_id", type=int, required=True, help="Process number")
run_args = parser.parse_args()

print(f"Running process {run_args.process_id}")

# Define fixed parameters
prop_to_use = 0.05
num_epochs = 100
batch_size = 3584
max_evals = 200
model_class = SCNN_TPL
apriori_relevance = torch.ones(801)
input_channels = 1
process_id = run_args.process_id
number_of_training_folds = 3
validation_test_folds = 2

train_fold_number = process_id // validation_test_folds
val_fold_number = process_id % validation_test_folds

# define an objective function
def objective(args):
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    scheduler_step_size = 10 #args['scheduler_step_size']
    scheduler_gamma = 0.5 #args['scheduler_gamma']
    leak_rate = args['leak_rate']
    dropout_rate = args['dropout_rate']
    entropy_weights = torch.tensor([args['entropy_weight_0'],args['entropy_weight_1'],args['entropy_weight_2'],args['entropy_weight_3']]).float()

    trainer = Trainer()
    train_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class, batch_size=batch_size, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds, training_fold_number=train_fold_number, val_fold_number = val_fold_number, early_stop=False,
              apriori_relevance = apriori_relevance, input_channels = input_channels, leak_rate = leak_rate, dropout_rate = dropout_rate)
    print(train_score)
    return {"loss": train_score[0], "status": STATUS_OK}

# define a search space
space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
    "weight_decay": hp.loguniform("weight_decay", np.log(1E-5), 0.2),
    # "scheduler_step_size": hp.quniform("scheduler_step_size", 1, 50, 1),
    # "scheduler_gamma": hp.uniform("scheduler_gamma", 0, 1),
    "leak_rate": hp.loguniform("leak_rate", np.log(1E-2), 0.3),
    "dropout_rate": hp.uniform("dropout_rate", 0.25, 0.95),
    "entropy_weight_0": hp.uniform("entropy_weight_0", 0.1, 1),
    "entropy_weight_1": hp.uniform("entropy_weight_1", 0.1, 1),
    "entropy_weight_2": hp.uniform("entropy_weight_2", 0.1, 1),
    "entropy_weight_3": hp.uniform("entropy_weight_3", 0.1, 1)
}

trials = Trials()

algorithm = atpe.suggest

# minimize the objective over the space
best = fmin(objective, space, algo=algorithm, max_evals=max_evals, trials=trials)

print(best)
print(space_eval(space, best))
save_directory = "/local/scratch/Data/TROPHY/Models/"
np.save(save_directory + "best_" + model_class.name + str(train_fold_number), best )
np.save(save_directory + "trials_" + model_class.name + str(train_fold_number), trials)