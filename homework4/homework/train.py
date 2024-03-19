import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim

## Taken from tests.py file for AP calculations ##
def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d ** 2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (abs(torch.min(px1, x1) - torch.max(px0, x0)) * abs(torch.min(py1, y1) - torch.max(py0, y0))) / \
          (abs(torch.max(px1, x1) - torch.min(px0, x0)) * abs(torch.max(py1, y1) - torch.min(py0, y0)))
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        lbl = torch.as_tensor(lbl.astype(float), dtype=torch.float32).view(-1, 4)
        d = torch.as_tensor(d, dtype=torch.float32).view(-1, 5)
        all_pair_is_close = self.is_close(d[:, 1:], lbl)

        # Get the box size and filter out small objects
        sz = abs(lbl[:, 2]-lbl[:, 0]) * abs(lbl[:, 3]-lbl[:, 1])

        # If we have detections find all true positives and count of the rest as false positives
        if len(d):
            detection_used = torch.zeros(len(d))
            # For all large objects
            for i in range(len(lbl)):
                if sz[i] >= self.min_size:
                    # Find a true positive
                    s, j = (d[:, 0] - 1e10 * detection_used - 1e10 * ~all_pair_is_close[:, i]).max(dim=0)
                    if not detection_used[j] and all_pair_is_close[j, i]:
                        detection_used[j] = 1
                        self.det.append((float(s), 1))

            # Mark any detection with a close small ground truth as used (no not count false positives)
            detection_used += all_pair_is_close[:, sz < self.min_size].any(dim=1)

            # All other detections are false positives
            for s in d[detection_used == 0, 0]:
                self.det.append((float(s), 0))

        # Total number of detections, used to count false negatives
        self.total_det += int(torch.sum(sz >= self.min_size))


    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        import numpy as np
        pr = np.array(self.curve, np.float32)
        return np.mean([np.max(pr[pr[:, 1] >= t, 0], initial=0) for t in np.linspace(0, 1, n_samples)])


## Helper functions ##
def calculate_ap(model, data_loader, device):
    pr_instances = [PR() for _ in range(3)] # For 3 classes...
    with torch.no_grad():
        for images, annotations in data_loader:
            images = images.to(device)
            detections = model.detect(images)
            for i, pr in enumerate(pr_instances):
                pr.add(detections[i], annotations[i].numpy())
    # Calculate AP for each class
    ap_values = [pr.average_prec for pr in pr_instances]
    return ap_values


def calculate_pos_weights(data_loader, device, q=0.66):
    # Initialize sums for calculating positive weights for each channel
    pos_sums = torch.zeros(3, device = device)  #3 heatmap channels
    total_pixels = torch.zeros(3, device = device)
    
    for imgs, annotations in data_loader:
        labels = annotations[0].to(device)
        for c in range(3):  # Iterate over each channel
            channel_labels = labels[:, c, :, :]
            above_threshold = (channel_labels > q).float()
            pos_sums[c] += above_threshold.sum()
            total_pixels[c] += torch.numel(channel_labels)
    
    # Calculate weights
    neg_counts = total_pixels - pos_sums
    pos_weights = neg_counts / pos_sums
    return pos_weights


def compute_loss(predictions, annotations, bce_loss, size_loss, device):
 
    # Get predictions
    heatmap_preds = predictions[:, :3, :, :]
    size_preds = predictions[:, :3, :, :].reshape(-1,2)

    # Get annotations
    heatmap_annotations = annotations[0].to(device)
    size_annotations = annotations[1].to(device).reshape(-1,2)

    # Calculate object centers
    object_centers = heatmap_annotations.sum(1, keepdim = True) > 0
    object_centers = object_centers.repeat(1, 2, 1, 1).reshape(-1, 2)
    
    # Get predictions and annotations for object centers
    size_preds_filtered = size_preds[object_centers[:, 0]]
    size_annotations_filtered = size_annotations[object_centers[:, 0]]

    # Calculate loss
    heatmap_loss_value = bce_loss(heatmap_preds, heatmap_annotations)
    size_loss_value = size_loss(size_preds_filtered, size_annotations_filtered)

    # Combine total loss (heatmap and size)
    combined_loss = heatmap_loss_value + size_loss_value
    return combined_loss


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

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    # Set up data transformation
    transformation = dense_transforms.Compose([
        dense_transforms.ToHeatmap(),
        dense_transforms.ToTensor()
    ])

    # Load in data
    train_data = load_detection_data('dense_data/train', transform = transformation, batch_size = args.batch_size)
    valid_data = load_detection_data('dense_data/valid', transform = transformation, batch_size = args.batch_size)

    # Loss functions
    initial_pos_weights = calculate_pos_weights(train_data).to(device)
    heatmap_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight = initial_pos_weights)
    size_loss_function = torch.nn.MSELoss()

    # Initialize tb logging
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Initialize best AP (average AP) value
    best_ap = 0.0
    global_step = 0

    # Training loop
    for epoch in range(args.epochs):
        
        # Set model to train
        model.train()
        for images, annotations in train_data:

            # Send to device
            images = images.to(device)
            annotations = [ann.to(device) for ann in annotations]

            # Zero gradient
            optimizer.zero_grad()

            # Get predictions (forward pass)
            predictions = model(images)

            # Calculate loss
            loss = compute_loss(predictions, annotations, heatmap_loss_function, size_loss_function, device)

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
        ap_values = calculate_ap(model, valid_data, device)
        average_ap = np.mean(ap_values)
        if average_ap > best_ap:
            best_ap = average_ap
            save_model(model)
            print(f"Saved model with new best avg AP: {best_ap: .4f}")
            print(f"Current AP values: {ap_values:.4f}")
        
        # Update pos_weights
        updated_pos_weights = calculate_pos_weights(train_data).to(device)
        heatmap_loss_function.pos_weight = updated_pos_weights

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, New avg AP: {average_ap: .4f}, Updated Weights: {updated_pos_weights}")



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
    args = parser.parse_args()
    train(args)
