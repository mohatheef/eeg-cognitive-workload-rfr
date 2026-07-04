# EEG-Based Cognitive Workload Estimation using Random Forest Regression

This repository contains the official Python implementation used in the peer-reviewed journal article:

**Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency Features from EEG Epochs with Random Forest Regression**

Published in *Scientific Reports* (Nature Portfolio, Q1 Journal), 2026  
DOI: [https://doi.org/10.1038/s41598-026-35986-5](https://doi.org/10.1038/s41598-026-35986-5)  
Zenodo Archive: [https://doi.org/10.5281/zenodo.18096105](https://doi.org/10.5281/zenodo.18096105)

This work presents an interpretable and computationally efficient machine learning framework for estimating cognitive workload from EEG signals during robot-assisted surgery.

---

## Overview

The proposed pipeline includes:
* **EEG Preprocessing**: Common Average Reference (CAR) filtering, notch filtering at 60 Hz, band-pass filtering (0.2–249 Hz), downsampling from 500 Hz to 128 Hz, and ICA-based artifact removal (with EOG channels dropped).
* **Feature Extraction**: Epoch-based feature extraction (1-second windows) for:
  * **Time-domain features**: mean, variance, skewness, kurtosis, root mean square (RMS), and zero crossings.
  * **Frequency-domain features**: band power (delta, theta, alpha, beta, gamma) using Welch's Power Spectral Density (PSD).
  * **Anatomical Mapping**: Automatic channel-to-lobe assignment based on 10-5 system prefixes (`Frontal`, `Temporal`, `Parietal`, `Occipital`).
* **Modeling & Benchmarking**: Random Forest Regression (RFR) compared against baseline regression models (SVR, Linear Regression, and XGBoost) using:
  * 80/20 train-test split.
  * 10-fold cross-validation.
  * Paired t-tests on absolute error distributions for statistical significance testing.
* **Visualization**: Generation of feature importance, correlation heatmaps, predicted vs. actual scatter plots, and regional metric comparisons.

---

## Repository Structure

```text
├── preprocessing/
│   └── eeg_preprocessing.py                 # CAR, filtering, downsampling, and ICA
├── feature_extraction/
│   └── epoch_feature_extraction.py           # Time/frequency extraction and lobe mapping
├── modeling/
│   ├── rfr_train_test_split.py              # Train-test benchmarking vs baseline models
│   └── rfr_10fold_cv.py                     # 10-fold cross-validation and paired t-tests
├── visualization/
│   └── cortical_region_visualization.py     # Plots recreating paper Figures 9, 10, 11, 12
├── verify_pipeline.py                       # End-to-end pipeline verification test
├── CITATION.cff                             # Citation metadata
├── requirements.txt                         # Dependency requirements list
└── README.md                                # Project documentation
```

---

## Installation

1. Clone this repository to your local system:
   ```bash
   git clone https://github.com/mohatheef/eeg-cognitive-workload-rfr.git
   cd eeg-cognitive-workload-rfr
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Availability

The dataset used in this study is publicly available on PhysioNet:
> Shafiei, S. B., et al. (2023).  
> *Electroencephalogram and Eye-Gaze Datasets for Robot-Assisted Surgery Performance Evaluation* (v1.0.0).  
> PhysioNet. [https://doi.org/10.13026/qj5m-n649](https://doi.org/10.13026/qj5m-n649)

To run the pipeline, download the EDF files from PhysioNet and place them in a raw data directory (e.g., `data/raw_edf`).

---

## Usage Guide

All scripts are equipped with a Command Line Interface (CLI) to allow flexible directory configuration.

### 1. EEG Preprocessing
Loads raw `.edf` files, filters and downsamples the data to 128 Hz, drops EOG channels, removes ocular artifacts using ICA, and saves preprocessed signals as `.csv` files:
```bash
python preprocessing/eeg_preprocessing.py --data_dir data/raw_edf --output_dir data/preprocessed_csv
```

### 2. Feature Extraction
Extracts time-domain and frequency-domain statistics from 1-second epochs, maps each channel to its corresponding cortical lobe, and exports a unified feature file:
```bash
python feature_extraction/epoch_feature_extraction.py --input_dir data/preprocessed_csv --output_file data/features/epoch_features.csv
```

### 3. Model Benchmarking
Benchmarks Random Forest Regression against SVR, Linear Regression, and XGBoost using an 80/20 train-test split:
```bash
python modeling/rfr_train_test_split.py --feature_file data/features/epoch_features.csv
```

### 4. 10-Fold Cross-Validation
Performs 10-fold cross-validation across all models and calculates paired t-test statistics against the training-mean baseline:
```bash
python modeling/rfr_10fold_cv.py --feature_file data/features/epoch_features.csv
```

### 5. Generate Figures
Produces the paper-equivalent figures and saves them to the output folder (default: `data/results_plots`):
* `figure_9_feature_importance.png`: Feature importance bars for each region.
* `figure_10_correlation_matrix.png`: Heatmap of predictor correlations.
* `figure_11_predicted_vs_actual.png`: Scatter plots of predicted vs true workload values.
* `figure_12_performance_comparison.png`: Comparison of $R^2$, RMSE, and MAE across lobes.
```bash
python visualization/cortical_region_visualization.py --feature_file data/features/epoch_features.csv --output_dir data/results_plots
```

---

## Verification & Testing

To verify the correct installation of dependencies and the integrity of the code scripts, run the end-to-end verification script. This script automatically generates synthetic EEG raw EDF data, runs all pipeline scripts, and checks the correctness of the generated feature data and plots:
```bash
python verify_pipeline.py
```

---

## Citation

If you use this code or build upon this work, please cite our journal article:

```text
Atheef, M. G. A., & Powar, O. S. (2026).
Estimating cognitive workload in robot-assisted surgery using time and frequency features from EEG epochs with random forest regression.
Scientific Reports, 16, 7624.
https://doi.org/10.1038/s41598-026-35986-5
```

---

## Authors

* **Mohammed Atheef G A**  
  *Department of Biomedical Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE), India*
* **Dr. Omkar S. Powar** (Corresponding Author)  
  *Department of Biomedical Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE), India*

---

## License

This project is licensed under the MIT License. See `LICENSE` or the publication details for details.
