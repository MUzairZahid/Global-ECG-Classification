"""
model.py

Author: Muhammad Uzair Zahid
Created: 07/10/2021
Edited: 11/08/2024
Description: This module provides 1D-SelfONN model for global ECG classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastonn import SelfONN1d as SelfONN1dlayer


class SelfONN1DClassifier(nn.Module):
    def __init__(self, q):
        super(SelfONN1DClassifier, self).__init__()
        
        # Self-ONN 1D layers
        self.conv1 = SelfONN1dlayer(in_channels=9, out_channels=32, kernel_size=3, q=q)
        self.conv2 = SelfONN1dlayer(in_channels=32, out_channels=64, kernel_size=3, q=q)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

        # Pooling layers
        self.pooling1 = nn.MaxPool1d(kernel_size=5)
        self.pooling2 = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=68, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=3)

    def forward(self, x1, x2):
        # First Self-ONN layer with batch normalization and Tanh activation
        x1 = torch.tanh(self.bn1(self.conv1(x1)))
        x1 = self.pooling1(x1)
        
        # Second Self-ONN layer with batch normalization and Tanh activation
        x1 = torch.tanh(self.bn2(self.conv2(x1)))
        x1 = self.pooling2(x1)
        
        # Flatten the output for fully connected layer input
        x1 = x1.view(-1, 64)

        # Concatenate features from x1 and additional input x2
        x = torch.cat((x1, x2), dim=1)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
