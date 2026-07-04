import os
import shutil
import subprocess
import sys
import numpy as np
import mne


def create_mock_edf(file_path, ch_names, sfreq=500, duration_sec=30):
    """Creates a mock EEG EDF file with specified channel names."""
    n_samples = int(sfreq * duration_sec)
    n_channels = len(ch_names)
    
    # Generate random signal data
    data = np.random.randn(n_channels, n_samples) * 1e-6  # scale to microvolts
    
    # Create MNE info and raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Export to EDF
    mne.export.export_raw(file_path, raw, fmt='edf', overwrite=True, verbose=False)
    print(f"Created mock EDF: {file_path}")


def main():
    test_dir = "temp_test_pipeline"
    raw_dir = os.path.join(test_dir, "raw_edf")
    prep_dir = os.path.join(test_dir, "preprocessed_csv")
    feat_file = os.path.join(test_dir, "features", "epoch_features.csv")
    plot_dir = os.path.join(test_dir, "results_plots")
    
    # Clean and recreate directories
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Define channels representing all 4 regions + EOG channels
    mock_channels = [
        # Frontal Lobe
        "Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "FC1", "FC2", "Cz", "C3", "C4",
        # Temporal Lobe
        "FT7", "FT8", "T7", "T8", "TP7", "TP8",
        # Parietal Lobe
        "CP1", "CP2", "P3", "P4", "Pz",
        # Occipital Lobe
        "PO3", "PO4", "O1", "O2", "Oz",
        # EOG (To be dropped)
        "EEGHEOGRCPz", "EEGHEOGLCPz", "EEGVEOGUCPz", "EEGVEOGLCPz"
    ]
    
    # Generate two mock subject files (each 10 seconds of data at 500 Hz)
    create_mock_edf(os.path.join(raw_dir, "subject_01.edf"), mock_channels, sfreq=500, duration_sec=10)
    create_mock_edf(os.path.join(raw_dir, "subject_02.edf"), mock_channels, sfreq=500, duration_sec=10)
    
    print("\n--- Running Preprocessing ---")
    subprocess.run([
        sys.executable, "preprocessing/eeg_preprocessing.py",
        "--data_dir", raw_dir,
        "--output_dir", prep_dir
    ], check=True)
    
    # Verify preprocessed files exist
    assert os.path.exists(os.path.join(prep_dir, "processed_subject_01.csv"))
    assert os.path.exists(os.path.join(prep_dir, "processed_subject_02.csv"))
    print("Preprocessing verification: SUCCESS")
    
    print("\n--- Running Feature Extraction ---")
    subprocess.run([
        sys.executable, "feature_extraction/epoch_feature_extraction.py",
        "--input_dir", prep_dir,
        "--output_file", feat_file,
        "--epoch_len", "1.0",
        "--sfreq", "128"
    ], check=True)
    
    # Verify features file exists
    assert os.path.exists(feat_file)
    print("Feature extraction verification: SUCCESS")
    
    print("\n--- Running Modeling: Train-Test Split ---")
    subprocess.run([
        sys.executable, "modeling/rfr_train_test_split.py",
        "--feature_file", feat_file
    ], check=True)
    print("Modeling (Train-Test) verification: SUCCESS")
    
    print("\n--- Running Modeling: 10-Fold CV ---")
    subprocess.run([
        sys.executable, "modeling/rfr_10fold_cv.py",
        "--feature_file", feat_file
    ], check=True)
    print("Modeling (10-Fold CV) verification: SUCCESS")
    
    print("\n--- Running Visualization ---")
    subprocess.run([
        sys.executable, "visualization/cortical_region_visualization.py",
        "--feature_file", feat_file,
        "--output_dir", plot_dir
    ], check=True)
    
    # Verify plots exist
    assert os.path.exists(os.path.join(plot_dir, "figure_9_feature_importance.png"))
    assert os.path.exists(os.path.join(plot_dir, "figure_10_correlation_matrix.png"))
    assert os.path.exists(os.path.join(plot_dir, "figure_11_predicted_vs_actual.png"))
    assert os.path.exists(os.path.join(plot_dir, "figure_12_performance_comparison.png"))
    print("Visualization verification: SUCCESS")
    
    # Clean up test directories
    shutil.rmtree(test_dir)
    print("\n=================================")
    print("ALL PIPELINE STEPS RUN SUCCESSFULLY!")
    print("=================================")


if __name__ == "__main__":
    main()
