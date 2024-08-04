# Clinical Notes Analysis

## Overview

This repository contains code for predicting diabetes and cancer from clinical notes using BERT and KNN. The project aims to determine whether each note indicates that the patient has diabetes and/or cancer based on clinical snippets provided.

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>

2. Install the required packages:
   ```bash
   pip install -r requirements.txt


## Scripts 
```bash
1. Load and Preprocess Data

  This script loads and preprocesses the data.
  
  python scripts/load_and_preprocess.py


2.Train Models
  This script trains two separate models for cancer and diabetes prediction.

  python scripts/train_models.py

3. Predict with KNN
  This script uses the trained models to extract logits and then applies KNN for final prediction on the unlabeled data.

  python scripts/predict_with_knn.py 
