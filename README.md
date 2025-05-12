# Master's Project - Deep Learning Approaches for Label-Free Tumour Image Segmentation Spectroscopy
This repo contains code developed as of May 2025 to analyse IR spectra for cancer classification tasks.
## Overview
- Exploration files and jupyter notebooks can be found in the Develop folder.
  - [Controller.ipynb](https://github.com/THartigan/DL_Cancer_Annotations/blob/main/Develop/Controller.ipynb) contains development code for neural network training and evlaluation.
  - [11_Using_Crime.ipynb](https://github.com/THartigan/DL_Cancer_Annotations/blob/main/Develop/11_Using_Crime.ipynb) contains code for calculating LIME explanationsand clustering in the latent space of a VAE.
- The Models folder contains custom PyTorch Model classes.
- The Processing folder contains key code for processing and utility function for data analysis.
  - The Trainer class defined in [Train.py](https://github.com/THartigan/DL_Cancer_Annotations/blob/main/Processing/Trainer.py) underpins most other functionality in the repository.
- The Results folder contains classification prediction images and LIME explainability results.
- The Run folder contains utility functions for executing long-running processes.
