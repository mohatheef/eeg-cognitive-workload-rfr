
"""
EEG Preprocessing Pipeline for Cognitive Workload Estimation

This script implements the EEG preprocessing steps used in the study:
"Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency
Features from EEG Epochs with Random Forest Regression".

Preprocessing steps:
1. Common Average Referencing (CAR)
2. Notch filtering at 60 Hz
3. Band-pass filtering (0.2–249 Hz)
4. Artifact removal using Independent Component Analysis (ICA)

Note:
- Raw EEG data are NOT included in this repository.
- Users must download the dataset from PhysioNet:
  DOI: https://doi.org/10.13026/qj5m-n649
"""

import os
import argparse
import mne
import pandas as pd
from mne.preprocessing import ICA
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="EEG Preprocessing Pipeline for Cognitive Workload Estimation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw_edf",
        help="Path to directory containing EDF files downloaded from PhysioNet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed_csv",
        help="Path where preprocessed CSV files will be saved"
    )
    args = parser.parse_args()

    DATA_FOLDER = args.data_dir
    OUTPUT_FOLDER = args.output_dir

    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Raw data folder '{DATA_FOLDER}' does not exist.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    edf_files = sorted([
        f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".edf")
    ])

    if not edf_files:
        print(f"No EDF files found in '{DATA_FOLDER}'.")
        return

    for file_name in tqdm(edf_files, desc="Processing EDF files"):
        file_path = os.path.join(DATA_FOLDER, file_name)

        try:
            # Load EEG data
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # Store channel names
            channel_names = raw.info["ch_names"]

            # Exclude EOG channels (as stated in the manuscript page 7)
            eog_channels = ["EEGHEOGRCPz", "EEGHEOGLCPz", "EEGVEOGUCPz", "EEGVEOGLCPz"]
            raw.drop_channels([ch for ch in eog_channels if ch in channel_names], on_missing="ignore")

            # Step 1: Common Average Reference
            raw.set_eeg_reference(ref_channels="average", projection=True, verbose=False)

            # Step 2: Notch filter (60 Hz)
            raw.notch_filter(freqs=60, verbose=False)

            # Step 3: Band-pass filter (0.2–249 Hz)
            raw.filter(
                l_freq=0.2,
                h_freq=249,
                fir_design="firwin",
                filter_length="auto",
                verbose=False
            )

            # Step 3.5: Downsample from 500 Hz to 128 Hz (as stated in manuscript page 5)
            raw.resample(sfreq=128, verbose=False)

            # Step 4: ICA-based artifact removal
            ica = ICA(
                n_components=15,
                random_state=97,
                max_iter=800
            )
            ica.fit(raw, verbose=False)

            # NOTE:
            # Components excluded here are placeholders.
            # In practice, components should be selected
            # based on visual inspection or automated criteria.
            ica.exclude = [0, 1]
            raw = ica.apply(raw, verbose=False)

            # Convert to DataFrame
            data = raw.get_data()
            df = pd.DataFrame(data.T, columns=raw.info["ch_names"])

            # Save preprocessed data
            output_file = os.path.join(
                OUTPUT_FOLDER,
                f"processed_{os.path.splitext(file_name)[0]}.csv"
            )
            df.to_csv(output_file, index=False)

        except Exception as err:
            print(f"Error processing {file_name}: {err}")

    print(f"Preprocessing completed. Files saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
