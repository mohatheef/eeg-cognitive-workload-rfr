
"""
Epoch-Based EEG Feature Extraction

This script implements epoch-level time-domain and frequency-domain feature
extraction used in the study:

"Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency
Features from EEG Epochs with Random Forest Regression".

Features extracted per 1-second epoch:
- Time-domain: mean, variance, skewness, kurtosis, RMS, zero crossings
- Frequency-domain: band power (delta, theta, alpha, beta, gamma) using Welch PSD

Note:
- EEG data are NOT included in this repository.
- Input CSV files are assumed to be preprocessed EEG signals.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm


# =========================
# USER-DEFINED PARAMETERS
# =========================

# Path to directory containing preprocessed EEG CSV files
INPUT_DIR = "PATH_TO_PREPROCESSED_CSV"

# Output feature file
OUTPUT_FILE = "PATH_TO_OUTPUT/epoch_features.csv"

# Epoch configuration
EPOCH_LENGTH_SEC = 1      # seconds
SAMPLING_RATE_HZ = 128    # Hz


# =========================
# FEATURE FUNCTIONS
# =========================

def extract_time_features(epoch: np.ndarray) -> dict:
    """Extract time-domain features from an EEG epoch."""
    return {
        "mean": np.mean(epoch),
        "variance": np.var(epoch),
        "skewness": pd.Series(epoch).skew(),
        "kurtosis": pd.Series(epoch).kurt(),
        "rms": np.sqrt(np.mean(epoch ** 2)),
        "zero_crossings": ((epoch[:-1] * epoch[1:]) < 0).sum(),
    }


def extract_frequency_features(epoch: np.ndarray,
                              sampling_rate: int) -> dict:
    """Extract frequency-domain features using Welch PSD."""
    freqs, psd = welch(
        epoch,
        fs=sampling_rate,
        nperseg=len(epoch)
    )

    return {
        "delta_power": np.sum(psd[(freqs >= 0.5) & (freqs < 4)]),
        "theta_power": np.sum(psd[(freqs >= 4) & (freqs < 8)]),
        "alpha_power": np.sum(psd[(freqs >= 8) & (freqs < 13)]),
        "beta_power": np.sum(psd[(freqs >= 13) & (freqs < 30)]),
        "gamma_power": np.sum(psd[freqs >= 30]),
    }


# =========================
# MAIN PROCESSING
# =========================

all_features = []

print("Extracting epoch-based EEG features...")

for file_name in tqdm(os.listdir(INPUT_DIR)):

    if not file_name.lower().endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_DIR, file_name)

    try:
        # Load EEG data
        data = pd.read_csv(file_path)

        # Basic validation
        if data.shape[1] < 4:
            raise ValueError(
                f"{file_name} contains fewer than 4 channels. Skipping."
            )

        # Process each channel independently
        for channel in data.columns:

            signal = data[channel].values
            samples_per_epoch = int(EPOCH_LENGTH_SEC * SAMPLING_RATE_HZ)
            n_epochs = len(signal) // samples_per_epoch

            for epoch_idx in range(n_epochs):

                epoch = signal[
                    epoch_idx * samples_per_epoch:
                    (epoch_idx + 1) * samples_per_epoch
                ]

                time_feats = extract_time_features(epoch)
                freq_feats = extract_frequency_features(
                    epoch,
                    SAMPLING_RATE_HZ
                )

                features = {
                    **time_feats,
                    **freq_feats,
                    "channel": channel,
                    "epoch": epoch_idx,
                    "source_file": file_name
                }

                all_features.append(features)

    except Exception as err:
        print(f"Error processing {file_name}: {err}")


# =========================
# SAVE FEATURES
# =========================

features_df = pd.DataFrame(all_features)
features_df.to_csv(OUTPUT_FILE, index=False)

print(f"Feature extraction completed. Saved to: {OUTPUT_FILE}")
