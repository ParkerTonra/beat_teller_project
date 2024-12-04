import torch.nn as nn
import torch
from torch.utils.data import Dataset



class TellerNet(nn.Module):
    def __init__(self, input_channels=15, target_length=1292):
        super(TellerNet, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_channels)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate the output length after convolution and pooling
        conv_output_length = self._calculate_output_length(target_length)
        self.fc1 = nn.Linear(128 * conv_output_length, 256)
        self.fc2 = nn.Linear(256, 1)  # Output is a single value (scaled tempo)

    def _calculate_output_length(self, input_length):
        # Calculate the length after three pooling layers
        length = input_length
        for _ in range(3):
            length = length // 2  # MaxPool with kernel_size=2 halves the length
        return length

    def forward(self, x):
        x = self.input_bn(x)
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x 