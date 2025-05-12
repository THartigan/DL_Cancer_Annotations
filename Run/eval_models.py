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
from Processing.misc import calculate_performance_stats_by_class
import warnings
np.warnings = warnings
warnings.filterwarnings("ignore")

weights = np.load("/local/scratch/Data/TROPHY/Models/Run6/best_SCNN_TPL0.npy", allow_pickle=True).item()
apriori_relevance = torch.ones(801).to("cuda")

trainer = Trainer()
evaluation_model = SCNN_TPL(apriori_relevance, weights['leak_rate'], weights['dropout_rate'], 1)
# Core_CNN_TPL: "/local/scratch/Data/TROPHY/Models/5.92Core_CNN_TPL.pth"
outputs = []
for i in range(0, 10):
    total_eval_confusion_matrix, predictions, true_outputs, raw_inputs = trainer.evaluate_test_data(evaluation_model, "/local/scratch/Data/TROPHY/Models/5.328381SCNN_TPL.pth", 1, 4, 5120, fold_number = i)
    print(total_eval_confusion_matrix)
    outputs.append(total_eval_confusion_matrix)
    print(f"Fold {i} done")
    sensitivity, specificity, accuracy, fpr = calculate_performance_stats_by_class(total_eval_confusion_matrix)
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Accuracy: {accuracy}")
    print(f"FPR: {fpr}")

print(outputs)
np.save("/local/scratch/code/TROPHY/colon_data_analysis/CRIME/Results/SCNN_TPL_Folds_Confusions.npy", outputs)
# true_outputs_hist = np.histogram(true_outputs, bins = [0,1,2,3])
