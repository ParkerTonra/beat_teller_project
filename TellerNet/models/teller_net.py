import torch.nn as nn
import torch


class TellerNet(nn.Module):
    def __init__(self):
        super(TellerNet, self).__init__()
        
        # Convolutional layers for processing mel spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 2)
        
        # Additional layers
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Process spectral features
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 64)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Attention mechanism
        x, _ = self.attention(x, x, x)
        
        # Take mean over temporal dimension
        x = torch.mean(x, dim=1)
        
        # Final fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x
