import logging
import argparse
import pickle
from pathlib import Path
import numpy as np
import scipy.signal as sg
import wfdb
import pywt
from wfdb import processing
from functools import partial
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Utility function to normalize signal
normalize = partial(processing.normalize_bound, lb=-1, ub=1)

def preprocess_signal(signal, args):
    """Remove baseline wander and filter the signal."""
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * args.sampling_rate) - 1), int(0.6 * args.sampling_rate) - 1)
    return signal - baseline

def adjust_r_peaks(signal, r_peaks, args):
    """Adjust R-peaks to max point for alignment based on the tolerance window."""
    adjusted_peaks = []
    for r_peak in r_peaks:
        left = max(r_peak - int(args.tolerance * args.sampling_rate), 0)
        right = min(r_peak + int(args.tolerance * args.sampling_rate), len(signal))
        adjusted_peaks.append(left + np.argmax(signal[left:right]))
    return np.array(adjusted_peaks, dtype=int)

def calculate_wavelet_coeffs(signal, args):
    """Calculate continuous wavelet transform coefficients."""
    scales = pywt.central_frequency(args.wavelet_type) * args.sampling_rate / np.arange(11, 101, 10)
    coeffs, _ = pywt.cwt(signal, scales, args.wavelet_type, sampling_period=1. / args.sampling_rate)
    return coeffs

def process_record(record, args):
    """Process a single ECG record to extract signal, R-peaks, and wavelet coefficients."""
    try:
        logger.info(f"Processing record {record}")

        record_path = os.path.join(args.raw_data_path, record)
        # Load signal and annotations
        signal = wfdb.rdrecord(record_path, channels=[0]).p_signal[:, 0]
        annotation = wfdb.rdann(record_path, "atr")
        r_peaks, labels = annotation.sample, np.array(annotation.symbol)

        # Preprocess signal and filter non-beat labels
        filtered_signal = preprocess_signal(signal, args)
        r_peaks, labels = r_peaks[[i for i, label in enumerate(labels) if label not in args.invalid_labels]], labels[[i for i, label in enumerate(labels) if label not in args.invalid_labels]]

        # Align and normalize signal
        r_peaks = adjust_r_peaks(filtered_signal, r_peaks, args)
        normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

        # Convert labels to AAMI categories
        categories = [args.AAMI_categories.get(label, -1) for label in labels]

        # Calculate wavelet coefficients
        coeffs = calculate_wavelet_coeffs(normalized_signal, args)

        return {
            "record": record,
            "coeffs": coeffs,
            "signal": normalized_signal,
            "r_peaks": r_peaks,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error processing record {record}: {e}")
        return None

def process_records_sequentially(records, args):
    """Process multiple records sequentially."""
    results = []
    for record in records:
        result = process_record(record, args)
        if result:
            results.append(result)
    return results

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Process ECG records for classification")
    parser.add_argument(
        "--raw_data_path", 
        type=str, 
        default="./mit-bih-arrhythmia-database-1.0.0", 
        help="Path to the folder containing raw MITBIH data."
    )
    parser.add_argument(
        "--processed_data_path", 
        type=str, 
        default="./MITBIH_data_processed", 
        help="Path to save the processed data"
    )
    args = parser.parse_args()

    os.makedirs(args.processed_data_path, exist_ok = True)

    # Define other arguments and constants
    args.train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]

    args.test_records = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]

    args.sampling_rate = 360
    args.invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    args.tolerance = 0.05
    args.wavelet_type = "mexh"
    args.AAMI_categories = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }

    logger.info("Processing training records...")
    train_data = process_records_sequentially(args.train_records, args)

    logger.info("Processing test records...")
    test_data = process_records_sequentially(args.test_records, args)

    # Save processed data
    args.processed_data_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    with open(args.processed_data_path, "wb") as f:
        pickle.dump((train_data, test_data), f, protocol=4)

    logger.info("Processing complete!")
