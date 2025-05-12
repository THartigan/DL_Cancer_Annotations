import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
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
from Processing.misc import calculate_performance_stats_by_class

def get_sample_list(number_first = False):
    """
    Returns: a list of sample identifiers and their associated sample number. The order is reversed if number_first is True.
    """

    sample_list = np.load("/local/scratch/Data/TROPHY/numpy/sample_list.npy", allow_pickle=True).item()
    
    if number_first:
        keys = list(sample_list.keys())
        sample_list = {sample_list[key]: key for key in keys}
    return sample_list


def get_models_from_sweep(sweep_id, metric_name="val_overall_accuracy", 
                         threshold=50.0, entity="tjh200-university-of-cambridge", 
                         project="TROPHY", cache_dir="/local/scratch/models_cache"):
    """
    Retrieve all models from a wandb sweep that meet a certain performance threshold.
    Uses a persistent local cache to avoid re-downloading models.
    """
    import wandb
    import os
    import shutil
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize API
    api = wandb.Api()
    
    # Get all runs from the sweep
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    
    selected_models = []
    
    for run in runs:
        # Skip failed runs
        if run.state != "finished":
            continue
            
        # First check if there's a pre-computed best metric in the summary
        best_metric_name = f"best_{metric_name}"
        best_metric_value = run.summary.get(best_metric_name)
        
        if best_metric_value is None:
            # Try to get from history
            history = run.scan_history(keys=[metric_name])
            metrics = [row.get(metric_name) for row in history if metric_name in row]
            if metrics:
                best_metric_value = max(metrics)
        
        # Skip if we couldn't find the metric
        if best_metric_value is None:
            print(f"  Could not find {metric_name} for run {run.name}")
            continue
            
        # Check if it meets our threshold
        print(f"Run {run.name}: {best_metric_value}")
        if best_metric_value > threshold:
            print(f"  Selected: {best_metric_value} > {threshold}")
            
            run_id = run.id
            
            # Create a unique directory for this run in our cache
            run_cache_dir = os.path.join(cache_dir, f"{run_id}")
            
            # Find the model artifact
            model_path = None
            try:
                # First check if we already have this model in our cache
                cached_models = []
                if os.path.exists(run_cache_dir):
                    cached_models = [f for f in os.listdir(run_cache_dir) if f.endswith('.pth')]
                
                if cached_models:
                    # Use the cached model
                    model_path = os.path.join(run_cache_dir, cached_models[0])
                    print(f"  Using cached model: {model_path}")
                else:
                    # Download the model
                    for artifact in run.logged_artifacts():
                        if artifact.type == "model" and "best-model" in artifact.name:
                            print(f"  Downloading artifact: {artifact.name}")
                            # Download to temporary location
                            temp_dir = artifact.download()
                            
                            # Find the model file
                            model_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
                            if model_files:
                                # Create run directory if needed
                                os.makedirs(run_cache_dir, exist_ok=True)
                                
                                # Copy model file to our persistent cache
                                src_path = os.path.join(temp_dir, model_files[0])
                                dst_path = os.path.join(run_cache_dir, model_files[0])
                                shutil.copy2(src_path, dst_path)
                                model_path = dst_path
                                print(f"  Copied model to persistent cache: {model_path}")
                                break
                
                if model_path:
                    model_info = {
                        "run_id": run_id,
                        "run_name": run.name,
                        "model_path": model_path,
                        "config": run.config,
                        "metric_value": best_metric_value
                    }
                    selected_models.append(model_info)
                    print(f"  Added model: {model_path}")
                else:
                    print(f"  No model file found for run {run.name}")
                    
            except Exception as e:
                print(f"  Error processing run {run.name}: {e}")
    
    print(f"\nFound {len(selected_models)} models above threshold {threshold}%")
    return selected_models

def display_and_save_figures(sweep_id, threshold, training_fold_number, val_fold_number, number_of_training_folds, validation_test_folds, display=False):
    selected_models = get_models_from_sweep(sweep_id, "val_overall_accuracy", threshold)
    trainer = Trainer()
    # config = best_object['config']['model_config']
    apriori_relevance = torch.ones(801).to('cuda')
    # Core_CNN_TPL: "/local/scratch/Data/TROPHY/Models/5.92Core_CNN_TPL.pth"
    for eval_mode in ["train", "val", "test"]:
        total_eval_confusion_matrix, predictions, true_outputs, raw_inputs, coords, samples_nums, evaluated_sample_nums = trainer.chat_evaluate_multiple_models(SCNN_TPL3, selected_models, 1, 4, 5120, apriori_relevance, eval_on=eval_mode, training_fold_number=training_fold_number, val_fold_number=val_fold_number, number_of_training_folds=number_of_training_folds, validation_test_folds=validation_test_folds)
    
        stats = calculate_performance_stats_by_class(total_eval_confusion_matrix)

        total_eval_stats = {
            "Accuracy": stats[2],
            "Sensitivity": stats[0],
            "Specificity": stats[1],
            "False positive rate": stats[3],
            "Overall Accuracy": stats[4],
            "Confusion Matrix": total_eval_confusion_matrix,
        }

        results_base_dir = "/local/scratch/code/TROPHY/colon_data_analysis/CRIME/Results"
        sweep_results_dir = os.path.join(results_base_dir, f"{sweep_id}_{threshold}")

        # Create directory for the sweep (and the Figure directory if necessary)
        os.makedirs(results_base_dir, exist_ok=True)
        os.makedirs(sweep_results_dir, exist_ok=True)
        if not os.path.exists(os.path.join(sweep_results_dir, f"total_eval_stats_{eval_mode}.npy")):
            np.save(os.path.join(sweep_results_dir, f"total_eval_stats_{eval_mode}.npy"), total_eval_stats)

        samples_of_interest = evaluated_sample_nums

        samples: [Sample] = np.load("/local/scratch/Data/TROPHY/numpy/samples.npy", allow_pickle=True)

        for i, sample_of_interest in enumerate(samples_of_interest):
            # Set figure paths including sample number, evaluation type
            data_figure_path = os.path.join(sweep_results_dir, 
                                        f"sample_{sample_of_interest}_{eval_mode}_data.png")
            pred_figure_path = os.path.join(sweep_results_dir, 
                                        f"sample_{sample_of_interest}_{eval_mode}_predictions.png")
            
            # If both figures already exist, then skip
            if os.path.exists(data_figure_path) and os.path.exists(pred_figure_path):
                print(f"Figures for sample {sample_of_interest} already made, skipping")
                continue
            
            # Get the sample data
            sample_indices = np.where(samples_nums == sample_of_interest)[0]
            sample_coords = coords[sample_indices]
            sample_predictions = predictions[sample_indices]
            sample_true_outputs = true_outputs[sample_indices]
            sample_raw_inputs = raw_inputs[sample_indices]
            sample = samples[sample_of_interest]

            # Calculate views, could be interesting to use the highest scored CRIME channels for false colour visualisation
            data_view = sample.calculate_view_from_params(sample_coords, sample_raw_inputs, channel=50)
            predictions_view = sample.calculate_view_from_params(sample_coords, sample_predictions, channel=-1)

            # Plot and save data view
            plt.figure(figsize=(10, 8))
            plt.imshow(data_view)
            plt.colorbar()
            plt.title(f"Sample {sample_of_interest} - Raw Data (Channel 50) - {eval_mode} set")
            plt.savefig(data_figure_path, dpi=150, bbox_inches='tight')
            if display:
                plt.show()
            
            # Plot and save predictions view
            plt.figure(figsize=(10, 8))
            plt.imshow(predictions_view, cmap='jet')
            plt.colorbar()
            plt.title(f"Sample {sample_of_interest} - Predictions (Class {sample_true_outputs[0]+1}) - {eval_mode} set")
            plt.savefig(pred_figure_path, dpi=150, bbox_inches='tight')
            if display:
                plt.show()

def get_fold_numbers(dataset_config, validation_test_folds):
    """
    Returns the training and validation fold numbers based on the dataset configuration.
    """
    train_fold_number = dataset_config // validation_test_folds
    val_fold_number = dataset_config % validation_test_folds
    
    return train_fold_number, val_fold_number