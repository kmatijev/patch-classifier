"""
Simple CNN for patch-level binary classification (mitosis vs background)
Input: 128x128 patch
Output: probability of containing mitosis (0-1)
"""

import torch
import torch.nn as nn


class PatchClassifier(nn.Module):
    """
    CNN classifier for patch-level binary classification.
    Much simpler than U-Net since we only care about patch label, not pixel-level segmentation.
    """
    def __init__(self):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Encoder
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)  # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 512]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 1]
        
        return x
