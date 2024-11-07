"""
main.py

Author: Muhammad Uzair Zahid
Created: 07/10/2021
Edited: 11/08/2024
Description: This script trains a 1D Self-Operational Neural Network (SelfONN) for ECG classification,
             as presented in the paper "Global ECG Classification by Self-Operational Neural Networks with Feature Injection."
             It includes data loading, model training, evaluation, and model saving with support for custom arguments.
"""

import os
import shutil
import numpy as np
import torch
import pywt
from functools import partial
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EpochScoring, Initializer
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from helper_functions import load_data
from models import SelfONN1DClassifier

# Set random seed for reproducibility
torch.manual_seed(0)

def get_arguments():
    """Function to define and retrieve script arguments for flexibility."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a Self-ONN model for ECG classification.")
    parser.add_argument("--sampling_rate", type=int, default=360, help="ECG signal sampling rate.")
    parser.add_argument("--wavelet_type", type=str, default="mexh", help="Type of wavelet used for ECG processing.")
    parser.add_argument("--processed_data_path", type=str, default="./MITBIH_data_processed/mitbih_processed.pkl",
                        help="Path to the processed data file.")
    parser.add_argument("--q_parameter", type=int, default=3, help="Parameter 'q' for Self-ONN.")
    parser.add_argument("--save_model_dir", type=str, default="./saved_models", help="Directory to save trained models.")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate for the optimizer.")
    parser.add_argument("--max_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--step_size", type=int, default=7, help="Step size for learning rate scheduler.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler.")

    return parser.parse_args()

def main(args):
    # Calculate wavelet scales based on the central frequency and sampling rate
    scales = pywt.central_frequency(args.wavelet_type) * args.sampling_rate / np.arange(10, 101, 10)

    # Load training and testing data
    (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet_type=args.wavelet_type, scales=scales, sampling_rate=args.sampling_rate, processed_data_path=args.processed_data_path
    )
    print("Data loaded successfully!")

    # Set up callbacks for model training
    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=args.step_size, gamma=args.gamma),
        EpochScoring(scoring="accuracy", lower_is_better=False),
    ]

    # Initialize and configure the model with skorch wrapper
    net = NeuralNetClassifier(
        SelfONN1DClassifier(q=args.q_parameter),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=args.learning_rate,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_split=predefined_split(Dataset({"x1": x1_train, "x2": x2_train}, y_train)),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0
    )
    
    # Train the model
    net.fit({"x1": x1_train, "x2": x2_train}, y_train)

    # Evaluate the model on test data
    y_true, y_pred = y_test, net.predict({"x1": x1_test, "x2": x2_test})
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

    # Save the trained model parameters
    os.makedirs(args.save_model_dir, exist_ok=True)
    model_path = f"{args.save_model_dir}/model_{args.wavelet_type}.pkl"
    net.save_params(f_params=model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    args = get_arguments()
    main(args)
