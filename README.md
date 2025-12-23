# eeg-cognitive-workload-rfr



# EEG-Based Cognitive Workload Estimation using Random Forest Regression

This repository contains the Python implementation used in the study:

**“Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency Features from EEG Epochs with Random Forest Regression”**

The code implements an end-to-end pipeline for EEG-based cognitive workload estimation, including preprocessing, feature extraction, and regression modeling.

---

## Overview

The workflow includes:
- EEG preprocessing (filtering, downsampling, common average referencing, ICA-based artifact removal)
- Epoch-based feature extraction (time-domain and frequency-domain features)
- Power Spectral Density estimation using Welch’s method
- Random Forest Regression modeling
- Model evaluation using train–test split and 10-fold cross-validation

---

## Repository Structure

- `preprocessing/` – EEG cleaning and preparation scripts  
- `feature_extraction/` – Time- and frequency-domain feature computation  
- `modeling/` – Random Forest training, validation, and comparison with baseline models  
- `utils/` – Helper functions and visualization utilities  

---

## Data Availability

This repository does **not** include EEG or eye-gaze data.

The dataset used in this study is publicly available from PhysioNet:

Shafiei, S. B., et al. (2023). *Electroencephalogram and eye-gaze datasets for robot-assisted surgery performance evaluation* (version 1.0.0).  
PhysioNet. https://doi.org/10.13026/qj5m-n649

Users must comply with PhysioNet’s data usage requirements.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
