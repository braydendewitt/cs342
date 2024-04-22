import torch
import torch.nn as nn
import torch.nn.functional as F

class ImitationModel(nn.Module):
    def __init__(self):
        # Call super
        super(ImitationModel, self).__init__()
        self.norm = nn.BatchNorm1d(11) # Normalize inputs
        self.fc1 = nn.Linear(11, 64) # First layer of 11 features
        self.fc2 = nn.Linear(64, 128) # Hidden layer
        self.fc3 = nn.Linear(128, 64) # Hidden layer
        self.fc4 = nn.Linear(64, 3) # Output layer for 3 actions (accelerate, steer, brake)

    def forward(self, x):
        print("Input shape:", x.shape)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.norm(x) # Normalization
        x = F.relu(self.fc1(x)) # Apply ReLU
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # Get output
        return x
