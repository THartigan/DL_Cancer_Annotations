import xgboost as xgb
import numpy as np
import shap
import os

class XGB_Train():
    name = "XGB_Train"
    def __init__(self):
        
        if os.path.isdir("/local/scratch/Data/TROPHY/numpy"):
            print("On csvm5")
            self.n_jobs = 15
            self.save_dir = "/local/scratch/code/TROPHY/colon_data_analysis/CRIME/Results/"
        elif os.path.isdir("/local/scratch-3/tjh200/processed_trophy_data"):
            print("On Kiiara")
            self.n_jobs = 20
            self.save_dir = "/local/scratch-3/tjh-200/colon_data_analysis/CRIME/Results/"
        elif os.path.isdir("/local/scratch/data/TROPHY/numpy"):
            print("On PC")
            self.n_jobs = 10
            self.save_dir = "/local/scratch/data/TROPHY/numpy/"
        elif os.path.isdir("/Volumes/T7/scratch/data/TROPHY/numpy"):
            print("On Mac")
            self.n_jobs = 10
            self.save_dir = "/Volumes/T7/scratch/data/TROPHY/numpy/"
        

    def train(self, x_train, y_train, x_val, y_val):
        clf = xgb.XGBClassifier(n_jobs=self.n_jobs)
        print("Fitting")
        print(len(x_train), len(y_train))
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        return y_pred
        # explainer = shap.Explainer(clf)
        # shap_values = explainer(x_train).values
        # shap_values_summed = np.sum(np.mean(np.abs(shap_values), axis=0), axis=1)
        # sorted_idx = np.argsort(shap_values_summed)
        # print(sorted_idx)
        # print(shap_values_summed)
        # np.save(self.save_dir + "xgb_shap.npy", shap_values_summed)