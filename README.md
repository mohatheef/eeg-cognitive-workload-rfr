# EEG-Based Cognitive Workload Estimation using Random Forest Regression

This repository contains the official Python implementation used in the peer-reviewed journal article:

**Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency Features from EEG Epochs with Random Forest Regression**

Published in *Scientific Reports* (Nature Portfolio, Q1 Journal), 2026  
DOI: https://doi.org/10.1038/s41598-026-35986-5

This work presents an interpretable and computationally efficient machine learning framework for estimating cognitive workload from EEG signals during robot-assisted surgery.

---

## Overview

The proposed pipeline includes:

- EEG preprocessing (filtering, downsampling, referencing, artifact removal)
- Epoch-based feature extraction (1-second windows)
- Time-domain EEG features
- Frequency-domain features using Power Spectral Density (Welch’s method)
- Random Forest Regression for cortical region–wise workload estimation
- Model evaluation using train–test split and 10-fold cross-validation
- Statistical validation using paired t-tests
- Baseline model comparison (SVR, Linear Regression, XGBoost)

---

## Repository Structure

preprocessing/        EEG filtering, referencing, ICA artifact removal
feature_extraction/  Time-domain and frequency-domain feature computation
modeling/             Random Forest training, validation, and benchmarking
visualization/        Topographic maps, spectrograms, and performance plots


---

## Data Availability

This repository does **not** include EEG or eye-gaze data.

The dataset used in this study is publicly available from PhysioNet:

Shafiei, S. B., et al. (2023).  
*Electroencephalogram and Eye-Gaze Datasets for Robot-Assisted Surgery Performance Evaluation* (v1.0.0).  
PhysioNet. https://doi.org/10.13026/qj5m-n649

Users must comply with PhysioNet’s data usage, citation, and ethical requirements.

---

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt


Citation
If you use this code or build upon this work, please cite:

Atheef, M. G. A., & Powar, O. S. (2026).
Estimating cognitive workload in robot-assisted surgery using time and frequency features from EEG epochs with random forest regression.
Scientific Reports.
https://doi.org/10.1038/s41598-026-35986-5
