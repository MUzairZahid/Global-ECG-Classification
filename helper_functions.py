"""
helper_functions.py

Author: Muhammad Uzair Zahid
Created: 07/10/2021
Edited: 11/08/2024
Description: This module provides helper functions to prepare and load ECG data for training and testing,
             including data segmentation, feature extraction, and normalization.
"""

import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler


def prepare_ecg_data(data, wavelet_type, scales, sampling_period):
    """
    Prepares ECG data for model training by segmenting heartbeats, calculating RR intervals, and 
    extracting features from wavelet coefficients.

    Parameters:
        data (dict): A dictionary containing ECG record data with keys 'coeffs', 'r_peaks', 'categories', and 'record'.
        wavelet_type (str): Wavelet type used for the wavelet transform.
        scales (list): List of scales for wavelet transformation.
        sampling_period (float): Sampling period for the ECG signal.

    Returns:
        tuple: (x1, x2, y, groups) where:
            - x1: List of wavelet-scaled beats.
            - x2: List of RR interval features.
            - y: List of heartbeat categories.
            - groups: List of patient identifiers.
    """
    # Define segmentation interval around R-peak
    before, after = 90, 126

    # Retrieve necessary data
    coeffs = data["coeffs"]
    r_peaks, categories = data["r_peaks"], data["categories"]

    # Calculate average RR interval to normalize inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    # Initialize lists for features and labels
    x1, x2, y, groups = [], [], [], []

    # Process each heartbeat except the first two and last two
    for i in range(2, len(r_peaks) - 2):
        if categories[i] in {3, 4}:  # Skip undesired categories
            continue

        # Calculate local RR interval for recent beats
        local_RR = np.mean(np.diff(r_peaks[max(i - 10, 0):i + 1]))

        # Extract heartbeat segment around R-peak using wavelet coefficients
        beat_segment = coeffs[:, r_peaks[i] - before : r_peaks[i] + after]

        # Append wavelet-scaled heartbeat and RR interval features
        x1.append(beat_segment)
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,                    # Previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,                    # Post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # RR Interval Ratio
            local_RR - avg_rri                                        # Local RR Interval
        ])
        
        # Append labels and patient record identifier
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups


def load_data(wavelet_type, scales, sampling_rate, processed_data_path):
    """
    Loads and processes ECG data for training and testing, including feature extraction and normalization.

    Parameters:
        wavelet_type (str): Wavelet type used for continuous wavelet transform.
        scales (list): List of scales for wavelet transformation.
        sampling_rate (int): Sampling rate of the ECG signals.
        processed_data_path (str): Path to the pickle file containing the processed data.

    Returns:
        tuple: (train_data, test_data) where each is a tuple containing:
            - x1: Wavelet-scaled beats.
            - x2: RR interval features.
            - y: Labels for each heartbeat.
            - groups: Patient identifiers.
    """
    # Load data from pickle file
    with open(processed_data_path, "rb") as f:
        train_data, test_data = pickle.load(f)

    # Prepare training data
    x1_train, x2_train, y_train, groups_train = [], [], [], []
    for data in train_data:
        x1, x2, y, groups = prepare_ecg_data(data=data, wavelet_type=wavelet_type, scales=scales, sampling_period=1.0 / sampling_rate)
        x1_train.append(x1)
        x2_train.append(x2)
        y_train.append(y)
        groups_train.append(groups)

    # Concatenate and format training data
    x1_train = np.concatenate(x1_train, axis=0).astype(np.float32)
    x2_train = np.concatenate(x2_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # Prepare testing data
    x1_test, x2_test, y_test, groups_test = [], [], [], []
    for data in test_data:
        x1, x2, y, groups = prepare_ecg_data(data=data, wavelet_type=wavelet_type, scales=scales, sampling_period=1.0 / sampling_rate)
        x1_test.append(x1)
        x2_test.append(x2)
        y_test.append(y)
        groups_test.append(groups)

    # Concatenate and format testing data
    x1_test = np.concatenate(x1_test, axis=0).astype(np.float32)
    x2_test = np.concatenate(x2_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    # Normalize x2 data using RobustScaler for training and testing
    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    return (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test)
