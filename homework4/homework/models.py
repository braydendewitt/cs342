import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    
    # Send heatmap to GPU
    heatmap = heatmap.to(device)

    # Convert to order 4 tensor
    heatmap_tensor = heatmap[None, None]

    # Use max pooling for local maxima
    max_pooled_heatmap = F.max_pool2d(heatmap_tensor, kernel_size = max_pool_ks, stride = 1, padding = max_pool_ks//2)

    # Find local maxima
    local_maxima = (heatmap == max_pooled_heatmap.squeeze())

    # Identify peaks that do not meet minimum score
    scores = heatmap[local_maxima]
    indices = torch.nonzero(local_maxima, as_tuple = True)
    valid_scores = scores > min_score

    # Only keep ones that are greater than minimum score
    scores = scores[valid_scores]
    indices = torch.stack(indices, dim = -1)[valid_scores]

    # Check how many detections there are; if more than max allowed, only keep the top/best ones
    if scores.numel() == 0: # No detections
        return []
    
    if scores.size(0) > max_det: # If more than allowed
        top_scores, top_indices = torch.topk(scores, k = max_det)
        indices = indices[top_indices]
        scores = top_scores
    else: # Sort by score
        _, sorted_indices = scores.sort(descending = True)
        scores = scores[sorted_indices]
        indices = indices[sorted_indices]

    # Convert to coordinates and attach score with it
    peaks = [(float(score.item()), int(x.item()), int(y.item())) for score, (y,x) in zip(scores, indices)]

    # Return list of peaks
    return peaks

class CNNClassifier(nn.Module): # Taken from HW 3 master solution
    class Block(nn.Module):
        # Convolutional block with skip connections (taken from HW 3 Master Solution)
        def __init__(self, n_input, n_output, kernel_size = 3, stride = 2):

            super().__init__()

            # First convolution layer with batch norm.
            self.c1 = nn.Conv2d(n_input, n_output, kernel_size = kernel_size, padding = kernel_size//2, stride = stride, bias = False)
            self.b1 = nn.BatchNorm2d(n_output)

            # Second convolution layer with batch norm.
            self.c2 = nn.Conv2d(n_output, n_output, kernel_size = kernel_size, padding = kernel_size//2, bias = False)
            self.b2 = nn.BatchNorm2d(n_output)

            # Third convolution layer with batch norm.
            self.c3 = nn.Conv2d(n_output, n_output, kernel_size = kernel_size, padding = kernel_size//2, bias = False)
            self.b3 = nn.BatchNorm2d(n_output)

            # Skip connection
            self.skip = nn.Conv2d(n_input, n_output, kernel_size = 1, stride = stride)
        
        def forward(self,x):
            # Apply convolutions and skip connection, and activation function
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

class FCN(nn.Module): # Taken from HW 3 master solution

    class UpBlock(nn.Module):
        # Up-convolution block for decoder
        def __init__(self, n_input, n_output, kernel_size = 3, stride = 2):

            super().__init__()
            self.c1 = nn.ConvTranspose2d(n_input, n_output, kernel_size = kernel_size, padding = kernel_size//2, stride = stride, output_padding = 1)
        
        def forward(self, x):
            # Apply up-convolution with activation function
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=5, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)
    

class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super(Detector, self).__init__()

        # Intialize FCN (3 output classes, 5 channels each) and send to GPU
        self.fcn = FCN(layers = [16, 32, 64, 128], n_output_channels = 15, kernel_size = 3, use_skip = True)
        self.fcn.to(device)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        # Forward pass through FCN
        return self.fcn(x)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        # Send image to device
        image = image.to(device)
        
        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.forward(image) # Adds batch dimension
            predictions = torch.sigmoid(predictions) # Convert to probabilities

        # Initialize detections
        detections = [[] for _ in range(3)]

        # For each class...
        for i in range(3):
            # Adjust indices
            heatmap_channel = i*5
            width_channel = heatmap_channel + 3
            height_channel = heatmap_channel + 4

            # Get heatmap and corresponding peaks
            heatmap = predictions[0, heatmap_channel]
            #print("Heatmap: ", heatmap)
            peaks = extract_peak(heatmap, max_det = 30)

            # Get detections
            for score, cx, cy in peaks:
                width = 0
                #width = predictions[0, width_channel, cy, cx].item() * image.shape[2]
                height = 0
                #height = predictions[0, height_channel, cy, cx].item() * image.shape[1]
                detections[i].append((score, cx, cy, width, height))
                #print("Detections i: ", detections[i])
        
        return detections

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
