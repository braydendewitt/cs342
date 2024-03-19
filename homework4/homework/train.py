import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim

# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2.0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction = 'none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


## Training loop ##
def train(args):
    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    from os import path

    # Set up device/GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # Initialize model
    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    # Set up data transformation
    transformation = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(flip_prob = 0.5),
        dense_transforms.ColorJitter(brightness = (0.7), contrast = (0.8), saturation = (0.7), hue = (0.2)),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])

    # Initialize focal loss
    focal_loss_function = FocalLoss(alpha = 0.25, gamma = 2.0, reduction = 'mean').to(device)
    
    # Initialize size loss
    size_loss_function = torch.nn.MSELoss().to(device)

    # Load in data
    train_data = load_detection_data('dense_data/train', transform = transformation, batch_size = args.batch_size)
    #valid_data = load_detection_data('dense_data/valid', transform = transformation, batch_size = args.batch_size)

    # Initialize tb logging
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Initialize loss
    current_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(args.epochs):
        
        # Set model to train
        model.train()
       
        for images, heatmaps, sizes in train_data:

            # Send to device
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            sizes = sizes.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Get predictions (forward pass)
            predictions = model(images)

            # Calculate loss
            heatmap_loss = focal_loss_function(predictions[:, :3, :, :], heatmaps)
            size_loss = size_loss_function(predictions[:, 3:5, :, :].permute(0, 2, 3, 1).reshape(-1, 2), sizes.reshape(-1, 2))
            loss = heatmap_loss + size_loss

            # Backward pass
            loss.backward()

            # Step in optimizer
            optimizer.step()

            # Log training loss in tb
            if train_logger:
                train_logger.add_scalar('loss', loss.item(), global_step = global_step)
            global_step += 1
        
        # Validate
        model.eval()

        # Calculate AP values
     
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Save model
        if float(loss.item()) < current_loss:
            current_loss = float(loss.item())
            save_model(model)
            print(f"Saving model at epoch {epoch+1} with loss of {loss.item()}")


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default = None)
    # Put custom arguments here
    parser.add_argument('--epochs', type=int, default=10) # Number of epochs
    parser.add_argument('--batch_size', type=int, default=32) # Batch size
    parser.add_argument('--lr', type=float, default=0.001) # Learning rate
    parser.add_argument('-c', '--continue_training', action='store_true') # Continue training
    args = parser.parse_args()
    train(args)
