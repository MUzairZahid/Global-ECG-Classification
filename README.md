# Global ECG Classification by Self-Operational Neural Networks with Feature Injection

## Overview

This repository implements an approach for global ECG classification using Self-Operational Neural Networks (Self-ONNs) with feature injection, designed to classify arrhythmias using the MIT-BIH dataset. Our model uses temporal and morphological features from ECG signals, achieving high accuracy while maintaining a compact and computationally efficient architecture.

The method outlined in this repository is based on the paper:
> **Global ECG Classification by Self-Operational Neural Networks with Feature Injection**  
> _Muhammad Uzair Zahid, Serkan Kiranyaz, and Moncef Gabbouj_  
> IEEE Transactions on Biomedical Engineering, Vol. 70, No. 1, January 2023.

### Project Structure
1. **Data Preprocessing**: `ecg_data_processing.py` — Processes raw ECG signals, aligns R-peaks, calculates wavelet coefficients, and generates datasets for model training and testing.
2. **Model Training**: `main.py` — Implements the 1D Self-ONN model with feature injection and trains it on preprocessed data.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Code](#running-the-code)
- [Proposed Approach](#proposed-approach)
- [Results](#results)
- [Contact and Collaboration](#contact-and-collaboration)
- [References](#references)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MUzairZahid/Global-ECG-Classification-by-Self-Operational-Neural-Networks-With-Feature-Injection.git
   cd yourrepo
   ```

2. **Download the MIT-BIH dataset:**
   Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

3. **Install dependencies:**
   - Install Python packages.
   - Install the **fastonn** library for Self-ONNs:
     ```bash
     git clone https://github.com/junaidmalik09/fastonn
     cd fastonn
     pip install .
     ```

---

## Data Preparation

1. **Preprocess the ECG data:**
   Use `ecg_data_processing.py` to preprocess the raw ECG data, align R-peaks, compute wavelet coefficients, and create training/testing datasets.

   ```bash
   python ecg_data_processing.py --raw_data_dir ./mit-bih-arrhythmia-database-1.0.0 --processed_data_dir ./MITBIH_data_processed
   ```

   This will save the processed data as `mitbih_processed.pkl` in the `processed_data_dir` directory.

---

## Running the Code

1. **Train the Model:**
   Run `main.py` after preprocessing the data. Specify paths and parameters as required.
   ```bash
   python main.py --wavelet_type "mexh" --sampling_rate 360 --q 3 --processed_data_path ./MITBIH_data_processed/mitbih_processed.pkl --save_model_dir ./saved_models
   ```

   ### Arguments for `main.py`:
   - `wavelet_type`: Type of wavelet to use (default: "mexh").
   - `sampling_rate`: Sampling rate of ECG signals (default: 360 Hz).
   - `q`: Degree of non-linearity for Self-ONN (default: 3).
   - `processed_data_path`: Path to the preprocessed data.
   - `save_model_dir`: Directory to save the trained model.

---

## Proposed Approach

The proposed 1D Self-ONN model effectively captures the morphological and temporal features of ECG signals. Key components of the approach include:

- **Self-ONN Layers**: These layers extract morphological features from individual ECG beats, enabling the model to differentiate arrhythmia types.
- **Feature Injection**: Temporal features based on R-R intervals are directly injected into the Self-ONN model, enriching the feature space for classification and providing critical information on the sequence and timing of beats.

> ![Figure 1: Proposed Approach.](figures/proposed_approach.png) 
> *Figure 1: The proposed approach and model architecture for classiﬁcation of ECG signals.*

---

## Results

The proposed approach demonstrates high accuracy in arrhythmia classification. The main results, as shown in Table 1, provide a comprehensive performance comparison across arrhythmia types.

> ![Table 1: Classification Performance](figures/results.png)  
> *Table 1: CLASSIFICATION PERFORMANCE OF THE PROPOSED 1D SELF-ONN WITH Q =3 AND FIVE COMPETING ALGORITHMS.

These results demonstrate that our Self-ONN approach can perform high-accuracy arrhythmia classification with minimal computational overhead, making it feasible for real-time, low-power ECG monitoring devices.

---

## Contact and Collaboration

For any questions, issues, or potential collaboration inquiries, please contact:

**Muhammad Uzair Zahid**  
Email: [muhammaduzair.zahid@tuni.fi](mailto:muhammaduzair.zahid@tuni.fi)  
LinkedIn: [https://www.linkedin.com/in/uzair-zahid/](https://www.linkedin.com/in/uzair-zahid/)
---


## References

1. Zahid, M. U., Kiranyaz, S., & Gabbouj, M. (2023). "Global ECG Classification by Self-Operational Neural Networks With Feature Injection." *IEEE Transactions on Biomedical Engineering*, 70(1), 205–214.
2. MIT-BIH Arrhythmia Database. Available at: [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
3. FastONN: GPU-based library for Operational Neural Networks. Available at: [https://github.com/junaidmalik09/fastonn](https://github.com/junaidmalik09/fastonn)
