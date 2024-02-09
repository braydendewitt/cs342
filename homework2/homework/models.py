import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """

        # Call parent init
        super(CNNClassifier, self).__init__()

        # First conv. layer (3 input channels, 32 output, 3x3 conv, stride and pad of 1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Second conv. layer (32 input, 64 output, 3x3 conv, stride and pad of 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Max pool layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Linear layer to transform to 128-dimensions (from 64*16*16 from above)
        self.linear1 = nn.Linear(in_features=64 * 16 * 16, out_features=128)

        # Second linear layer (transforms from 128 to 6 dimensions)
        self.linear2 = nn.Linear(in_features=128, out_features=6)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        
        # Apply first conv., apply ReLU and max pool
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second conv., apply ReLU and max pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten output before linear layers
        x = x.view(-1, 64*16*16)

        # Apply first linear layer
        x = F.relu(self.linear1(x))

        # Apply second linear layer to get outputs
        x = self.linear2(x)

        # Return outputs (as tensor)
        return x


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
