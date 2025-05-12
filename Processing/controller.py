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
from Processing.Trainer import Trainer
from hyperopt import hp, fmin, atpe, space_eval, STATUS_OK, Trials
from functools import partial
import warnings
np.warnings = warnings
warnings.filterwarnings("ignore")

# Define fixed parameters
prop_to_use = 1
num_epochs = 10
model_class = Core_CNN
apriori_relevance = torch.ones(801)
input_channels = 1

# define an objective function
def objective(args):
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    scheduler_step_size = args['scheduler_step_size']
    scheduler_gamma = args['scheduler_gamma']
    leak_rate = args['leak_rate']
    dropout_rate = args['dropout_rate']
    entropy_weights = torch.tensor([args['entropy_weight_0'],args['entropy_weight_1'],args['entropy_weight_2'],args['entropy_weight_3']]).float()

    trainer = Trainer()
    train_score = trainer.train(prop_to_use, learning_rate, weight_decay, scheduler_step_size, scheduler_gamma, num_epochs, entropy_weights, model_class,
              apriori_relevance = apriori_relevance, input_channels = input_channels, leak_rate = leak_rate, dropout_rate = dropout_rate)
    print(train_score)
    return {"loss": train_score, "status": STATUS_OK}

previous_best = np.load("/local/scratch/Data/TROPHY/Models/best.npy", allow_pickle = True).item()
print(previous_best)

# define a search space
space = {
    "learning_rate": hp.normal("learning_rate", previous_best["learning_rate"], 1e-5),
    "weight_decay": hp.normal("weight_decay", previous_best["weight_decay"], 0.01),
    "scheduler_step_size": previous_best["scheduler_step_size"],
    "scheduler_gamma": hp.normal("scheduler_gamma", previous_best["scheduler_gamma"], 0.1),
    "leak_rate": hp.normal("leak_rate", previous_best["leak_rate"], 0.02),
    "dropout_rate": hp.normal("dropout_rate", previous_best["dropout_rate"], 0.05),
    "entropy_weight_0": hp.normal("entropy_weight_0", previous_best["entropy_weight_0"], 0.5),
    "entropy_weight_1": hp.normal("entropy_weight_1", previous_best["entropy_weight_1"], 0.5),
    "entropy_weight_2": hp.normal("entropy_weight_2", previous_best["entropy_weight_2"], 0.5),
    "entropy_weight_3": hp.normal("entropy_weight_3", previous_best["entropy_weight_3"], 0.5)
}

trials = Trials()

algorithm = atpe.suggest

# minimize the objective over the space
best = fmin(objective, space, algo=algorithm, max_evals=250, trials=trials)

print(best)
print(space_eval(space, best))
save_directory = "/local/scratch/Data/TROPHY/Models/"
np.save(save_directory + "best2", best)
np.save(save_directory + "trials2", trials)