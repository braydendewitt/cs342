import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim

## Helper functions ##
def calculate_pos_weights(data_loader, device, q=0.5):
    # Data structure to store the sum of weights for each channel
    weights_sum = torch.zeros(3, device=device)
    # Counter for the number of batches, to calculate the average later
    batch_count = 0
    
    for imgs, heatmaps, sizes in data_loader:
        batch_size = imgs.shape[0]
        for c in range(3):  # Iterate over each channel
            channel_labels = heatmaps[:, c, :, :]
            
            # Calculate the denominator: count of values above the threshold q
            denominator = (channel_labels > q).sum().float()
            # Calculate the numerator: total possible values - denominator
            numerator = (batch_size * channel_labels.shape[1] * channel_labels.shape[2]) - denominator
            
            # Calculate the weights for this batch and channel
            batch_weight = numerator / denominator if denominator != 0 else torch.tensor(0.0, device=device)
            
            # Aggregate the weights for averaging later
            weights_sum[c] += batch_weight
        
        batch_count += 1
    
    # Calculate the average weights across all batches
    avg_weights = weights_sum / batch_count
    
    return avg_weights

def compute_loss(predictions, annotations, bce_loss, size_loss, device):
 
    # Unpack tuple
    heatmap_annotations, size_annotations = annotations

    # Get predictions
    heatmap_preds = predictions[:, :3, :, :] # Use first 3 channels for heatmap
    size_preds = predictions[:, 3:5, :, :] # Use last two channels for size

    # Resize and flatten
    size_preds_flat = size_preds.permute(0,2,3,1).reshape(-1,2)
    size_annotations_flat = size_annotations.reshape(-1,2)

    # Calculate heatmap loss
    #print("heatmap_preds shape:", heatmap_preds.shape)
    #print("heatmap_annotations shape:", heatmap_annotations.shape)
    #print("pos_weight shape:", bce_loss.pos_weight.shape)

    bce_loss.pos_weight = bce_loss.pos_weight.to(device)
    heatmap_loss_value = bce_loss(heatmap_preds, heatmap_annotations)

    # Calculate object centers (for size predictions)
    object_centers = heatmap_annotations.sum(1, keepdim = True) > 0
    object_centers_flat = object_centers.view(-1) # Flatten to match with size

    # Filter to only object centers
    size_preds_filtered = size_preds_flat[object_centers_flat]
    size_annotations_filtered = size_annotations_flat[object_centers_flat]

    # Calculate size loss
    size_loss_value = size_loss(size_preds_filtered, size_annotations_filtered) if object_centers_flat.any() else 0

    # Combine total loss (heatmap and size)
    combined_loss = heatmap_loss_value + size_loss_value
    return heatmap_loss_value #### ONLY PREDICTING HEATMAP RIGHT NOW #####


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
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)

    # Set up data transformation
    transformation = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(flip_prob = 0.5),
        dense_transforms.ColorJitter(brightness = (0.7), contrast = (0.8), saturation = (0.7), hue = (0.2)),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])

    # Load in data
    train_data = load_detection_data('dense_data/train', transform = transformation, batch_size = args.batch_size)
    #valid_data = load_detection_data('dense_data/valid', transform = transformation, batch_size = args.batch_size)

    # Loss functions
    #initial_pos_weights = torch.tensor([1.0, 1.0, 1.0], device = device) # Initial pos_weights
    #initial_pos_weights = initial_pos_weights.reshape(1, -1, 1, 1) # Reshape for broadcasting
    #heatmap_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight = initial_pos_weights.to(device))
    weight_for_c0 = 1
    weight_for_c1 = 0.412/0.018
    weight_for_c2 = 0.412/0.018
    pos_weights = torch.tensor([weight_for_c0, weight_for_c1, weight_for_c2], device = device)
    pos_weights = pos_weights.reshape(1, -1, 1, 1)
    heatmap_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weights)
    size_loss_function = torch.nn.MSELoss()

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
            loss = compute_loss(predictions, (heatmaps, sizes), heatmap_loss_function, size_loss_function, device)

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
     

            
        # Update pos_weights
       # updated_pos_weights = calculate_pos_weights(train_data, device)
        #updated_pos_weights = updated_pos_weights.reshape(1, -1, 1, 1)
        #heatmap_loss_function.pos_weight = updated_pos_weights.to(device)
        #print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Updated Weights: {updated_pos_weights}")
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()},  Weights: {pos_weights}")

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

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--epochs', type=int, default=10) # Number of epochs
    parser.add_argument('--batch_size', type=int, default=32) # Batch size
    parser.add_argument('--lr', type=float, default=0.001) # Learning rate
    parser.add_argument('--wd', type=float, default=1e-4) # Weight decay
    parser.add_argument('-c', '--continue_training', action='store_true') # Continue training
    args = parser.parse_args()
    train(args)
