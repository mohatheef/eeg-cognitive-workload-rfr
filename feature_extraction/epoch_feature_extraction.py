
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
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm


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


def map_channel_to_cortical_area(channel_name: str) -> str:
    """Map standard 10-5/10-10 channel names to cortical regions."""
    ch = channel_name.upper().replace("EEG", "").replace("-", "").strip()

    # Exclude EOG channels
    if any(eog in ch for eog in ["EOG", "HEOG", "VEOG"]):
        return None

    # Prefix-based mapping rules derived from the 10-5 standard layout
    if ch.startswith("FT"):
        return "Temporal"
    elif ch.startswith("FC"):
        return "Frontal"
    elif ch.startswith("CP"):
        return "Parietal"
    elif ch.startswith("FP"):
        return "Frontal"
    elif ch.startswith("AF"):
        return "Frontal"
    elif ch.startswith("F"):
        return "Frontal"
    elif ch.startswith("C"):
        return "Frontal"  # Grouping Central channels under Frontal
    elif ch.startswith("TP") or ch.startswith("T"):
        return "Temporal"
    elif ch.startswith("PO"):
        return "Occipital"
    elif ch.startswith("P"):
        return "Parietal"
    elif ch.startswith("O") or ch.startswith("I"):
        return "Occipital"
    return None


# =========================
# MAIN PROCESSING
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Epoch-Based EEG Feature Extraction"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/preprocessed_csv",
        help="Path to directory containing preprocessed EEG CSV files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/features/epoch_features.csv",
        help="Path to save the generated epoch features CSV file"
    )
    parser.add_argument(
        "--epoch_len",
        type=float,
        default=1.0,
        help="Epoch duration in seconds"
    )
    parser.add_argument(
        "--sfreq",
        type=int,
        default=128,
        help="Sampling rate in Hz"
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_FILE = args.output_file
    EPOCH_LENGTH_SEC = args.epoch_len
    SAMPLING_RATE_HZ = args.sfreq

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input folder '{INPUT_DIR}' does not exist.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_features = []
    file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")])

    if not file_list:
        print(f"No preprocessed CSV files found in '{INPUT_DIR}'.")
        return

    print("Extracting epoch-based EEG features...")

    for file_name in tqdm(file_list, desc="Processing CSV files"):
        file_path = os.path.join(INPUT_DIR, file_name)

        try:
            # Load EEG data
            data = pd.read_csv(file_path)

            # Process each channel independently
            for channel in data.columns:
                # Map channel name to cortical region
                cortical_area = map_channel_to_cortical_area(channel)
                if cortical_area is None:
                    continue  # Skip EOG or unmapped channels

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
                        "cortical_area": cortical_area,
                        "epoch": epoch_idx,
                        "source_file": file_name
                    }

                    all_features.append(features)

        except Exception as err:
            print(f"Error processing {file_name}: {err}")

    # Save features if extracted
    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Feature extraction completed. Saved to: {OUTPUT_FILE}")
    else:
        print("No features extracted.")


if __name__ == "__main__":
    main()
