import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

# Encoder part of NN
class EncoderPortion(torch.nn.Module):
    # init
    def __init__(self, input_channels = 3):
        # Call super
        super().__init__()
        # First conv. layer
        self.conv1 = torch.nn.Conv2d(in_channels = input_channels, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        # Second conv. layer
        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        # Third conv. layer
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
    
    # forward
    def forward(self, x):
        # First conv. with Relu
        x = F.relu(self.conv1(x))
        # Second conv.
        x = F.relu(self.conv2(x))
        # Third conv.
        x = F.relu(self.conv3(x))
        # Output
        return x
    
# Decoder part of NN
class DecoderPortion(torch.nn.Module):
    # init
    def __init__(self, output_channels = 1):
        # Call super
        super().__init__()
        # Upsample
        self.conv1 = torch.nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        # Upsample again
        self.conv2 = torch.nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding = 1)
        # Reduce to 1 output
        self.conv3 = torch.nn.ConvTranspose2d(in_channels = 16, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1)
    
    # forward
    def forward(self, x):
        # First upsample
        x = F.relu(self.conv1(x))
        # Second upsample
        x = F.relu(self.conv2(x))
        # Third upsample
        x = F.relu(self.conv3(x))
        # Output
        return x

class Planner(torch.nn.Module):
    def __init__(self):
        super(Planner, self).__init__()

        """
        Your code here
        """

        # Call encoder and decoder
        self.encoder = EncoderPortion()
        self.decoder = DecoderPortion()
        

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        
        # Encode image
        encoded_image = self.encoder(img)
        # Get heatmap
        heatmap = self.decoder(encoded_image)
        heatmap = heatmap.squeeze(1)
        # Get peak of heatmap (aiming point)
        aim_point = spatial_argmax(heatmap)
        # Output aim point
        return aim_point


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
