
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
import mne
import pandas as pd
from mne.preprocessing import ICA
from tqdm import tqdm


# =========================
# USER-DEFINED PATHS
# =========================

# Path to directory containing EDF files downloaded from PhysioNet
DATA_FOLDER = "PATH_TO_EEG_EDF_FILES"

# Path where preprocessed CSV files will be saved
OUTPUT_FOLDER = "PATH_TO_OUTPUT_CSV"


# =========================
# CREATE OUTPUT DIRECTORY
# =========================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================
# LIST EDF FILES
# =========================
edf_files = sorted([
    f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".edf")
])


# =========================
# PROCESS EACH EDF FILE
# =========================
for file_name in tqdm(edf_files, desc="Processing EDF files"):

    file_path = os.path.join(DATA_FOLDER, file_name)

    # Load EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Store channel names
    channel_names = raw.info["ch_names"]

    # -------------------------
    # Step 1: Common Average Reference
    # -------------------------
    raw.set_eeg_reference(ref_channels="average", projection=True)

    # -------------------------
    # Step 2: Notch filter (60 Hz)
    # -------------------------
    raw.notch_filter(freqs=60)

    # -------------------------
    # Step 3: Band-pass filter (0.2–249 Hz)
    # -------------------------
    raw.filter(
        l_freq=0.2,
        h_freq=249,
        fir_design="firwin",
        filter_length="auto"
    )

    # -------------------------
    # Step 4: ICA-based artifact removal
    # -------------------------
    ica = ICA(
        n_components=15,
        random_state=97,
        max_iter=800
    )
    ica.fit(raw)

    # NOTE:
    # Components excluded here are placeholders.
    # In practice, components should be selected
    # based on visual inspection or automated criteria.
    ica.exclude = [0, 1]

    raw = ica.apply(raw)

    # -------------------------
    # (Optional) Surface Laplacian
    # -------------------------
    # Uncomment if electrode locations are available
    # montage = mne.channels.make_standard_montage("standard_1020")
    # raw.set_montage(montage)
    # raw = mne.preprocessing.compute_current_source_density(raw)

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    data = raw.get_data()
    df = pd.DataFrame(data.T, columns=channel_names)

    # -------------------------
    # Save preprocessed data
    # -------------------------
    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"processed_{os.path.splitext(file_name)[0]}.csv"
    )
    df.to_csv(output_file, index=False)


print(f"Preprocessing completed. Files saved to: {OUTPUT_FOLDER}")
