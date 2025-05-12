config = {
    "method": "random",
    "metric": {
        "goal": "minimize",
        "name": "val_loss"
    },
    "parameters": {
        "apriori_relevance": {
            "distribution": "categorical",
            "values": [
                "[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]"
            ]
        },
        "batch_size": {
            "value": 3584
        },
        "dropout_rate": {
            "distribution": "log_uniform",
            "max": 0.9,
            "min": 0.01
        },
        "early_stop": {
            "value": "false"
        },
        "input_channels": {
            "value": 1
        },
        "leak_rate": {
            "distribution": "log_uniform",
            "max": 0.8,
            "min": 1e-06
        },
        "learning_rate": {
            "distribution": "log_uniform",
            "max": 1,
            "min": 0.0001
        },
        "model_class": {
            "distribution": "categorical",
            "values": ["SCNN_TPL"]
        },
        "num_epochs": {
            "value": 100
        },
        "number_of_training_folds": {
            "value": 4
        },
        "prop_to_use": {
            "value": 0.05
        },
        "scheduler_gamma": {
            "distribution": "uniform",
            "max": 1,
            "min": 0.25
        },
        "scheduler_step_size": {
            "distribution": "int_uniform",
            "max": 20,
            "min": 5
        },
        "validation_test_folds": {
            "value": 2
        },
        "weight_decay": {
            "distribution": "log_uniform",
            "max": 0.5,
            "min": 1e-05
        }
    },
    "program": "train.py"
}