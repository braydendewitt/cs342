import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CNNResidualBlock(nn.Module):
    # Residual block with two convolutions and a skip connection
    def __init__(self, in_channels, out_channels, stride = 1):
        
        # Call parent
        super(CNNResidualBlock, self).__init__()

        # Initialize
        self.stride = stride
        adjusted_output_channels = out_channels // 2

        # First conv. layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, adjusted_output_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(adjusted_output_channels)

        # Second conv. layer with batch normalization
        self.conv2 = nn.Conv2d(adjusted_output_channels, adjusted_output_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(adjusted_output_channels)

        # Skip connection and adjust output for sizing issues
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != adjusted_output_channels:
            # Fix sizing/output issues if necessary
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, adjusted_output_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(adjusted_output_channels)
            )
        # Update dimensions after using torch.cat
        self.adjust_channels = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Apply first conv., batch normalization, and ReLU
        output = F.relu(self.bn1(self.conv1(x)))
        # Apply second conv., batch normalization
        output = self.bn2(self.conv2(output))
        # Skip connection
        shortcut = self.shortcut(x)
        output = torch.cat((output, shortcut), dim = 1)
        output = self.adjust_channels(output)
        # ReLU new output
        output = F.relu(output)
        # Return final output
        return output

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        
        # Call parent
        super(CNNClassifier, self).__init__()

        # Normalization layer
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        # First convolution and batch norm. - before the residual blocks
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(num_features = 64)

        # Residual blocks - reduces spatial dimension as it moves along
        self.block1 = CNNResidualBlock(64, 128, stride = 1)
        self.block2 = CNNResidualBlock(128, 256, stride = 2)
        self.block3 = CNNResidualBlock(256, 512, stride = 2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Classifier for the 6 classes
        self.linear = nn.Linear(512, 6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        
        # Normalize inputs
        x = self.normalize(x)

        # First convolution and layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Pool and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.linear(x)

        # Output
        return x



class FCN(torch.nn.Module):
    def __init__(self):
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        
        # Call super
        super(FCN, self).__init__()

        # Input normalization layer
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        # Encoder layers
        self.down1 = CNNResidualBlock(in_channels = 3, out_channels = 64, stride = 2)
        self.down2 = CNNResidualBlock(in_channels = 64, out_channels = 128, stride = 2)

        # Bridge between encoder and decoder layers
        self.bridge = CNNResidualBlock(in_channels = 128, out_channels = 128, stride = 1)

        # Decoder layers with spatial dimension adjustments
        self.up1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.adjust_up1_channels = nn.Conv2d(128, 64, kernel_size = 1) # Adjustment layer for after skip connection (adding d1 to u1)
        self.up2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.adjust_up2_channels = nn.Conv2d(35, 32, kernel_size = 1) # Adjustment layer for after skip connection (adding x to u2)

        # Output to the 5 classes
        self.final = nn.Conv2d(in_channels = 32, out_channels = 5, kernel_size = 1)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        # Get original image size
        original_image_size = (x.size(2), x.size(3))
        print(f"ORIGINAL SIZE: {x.shape}")

        # Determine if padding is necessary
        if x.size(2) < 128 or x.size(3) < 96:
            print("SMALL IMAGE")
            # Get pad height and width required
            pad_height = max(0, 128 - x.size(2))
            pad_width = max(0, 96-x.size(3))
            # Set all padding values
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            # Pad image
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode = 'constant', value = 0)
            print(f"PADDED INPUT: {x.shape}")

        
        # Normalization
        x = self.normalize(x)
        print(f"Normalized: {x.shape}")  

        # Encoder
        d1 = self.down1(x)
        print(f"D1: {d1.shape}")
        d2 = self.down2(d1)
        print(f"D2: {d2.shape}")

        # Bridge to decoder
        bridge = self.bridge(d2)
        print(f"Bridge: {bridge.shape}")

        # Decoder
        u1 = self.up1(bridge)
        print(f"U1: {u1.shape}")
        u1 = torch.cat((u1, d1[:, :, :u1.size(2), :u1.size(3)]), dim = 1) # Skip connection, ensure dimensions
        print(f"U1 after torch.cat: {u1.shape}")
        u1 = self.adjust_up1_channels(u1) # Adjust dimensions after torch.cat
        print(f"U1 after adjust: {u1.shape}")

        u2 = self.up2(u1)
        print(f"U2: {u2.shape}")
        u2 = torch.cat((u2, x[:, :, :u2.size(2), :u2.size(3)]), dim = 1) # Skip connection, ensure dimensions
        print(f"U2 after torch.cat: {u2.shape}")
        u2 = self.adjust_up2_channels(u2) # Adjust dimensions after torch.cat
        print(f"U2 after adjust: {u2.shape}")

        # Output
        output = self.final(u2)
        print(f"Output before crop: {output.shape}")
        output = output[:, :, :original_image_size[0], :original_image_size[1]] # Crop output to original image
        print(f"OUTPUT - after crop: {output.shape}")
        print(f"\n")
        return output


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
