import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim

# Detection Evaluation (for model evaluation and calculating AP scores)
# Taken from tests.py and edited to fit my training loop

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

class ModelDetectionEval:
    def __init__(self, model, device):
        # Initialize model, device, and stats for the 3 classes
        self.model = model.eval().to(device)
        self.device = device
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close = point_close) for _ in range(3)]
        self.pr_iou = [PR(is_close = box_iou) for _ in range(3)]

    def evaluate(self, dataset):
        # Run through dataset
        for img, *gts in dataset:
            with torch.no_grad():
                # Get detections
                detections = self.model.detect(img.to(self.device))
                # Add stats
                for i, gt in enumerate(gts):
                    self.pr_box[i].add(detections[i], gt)
                    self.pr_dist[i].add(detections[i], gt)
                    self.pr_iou[i].add(detections[i], gt)
    
    def calculate_ap_scores(self):
        # Calculate scores and return them as output
        ap_scores = {
            "box_ap": [pr.average_prec for pr in self.pr_box],
            "dist_ap": [pr.average_prec for pr in self.pr_dist],
            "iou_ap": [pr.average_prec for pr in self.pr_iou]
        }
        return ap_scores

def calculate_overall_ap(ap_scores):
    # Calculate overall AP
    total_ap = 0
    count = 0
    for scores in ap_scores.values():
        total_ap += sum(scores) # Sum all AP scores
        count += len(scores) # Get count of number of AP scores
    return total_ap / count if count > 0 else 0


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
    train_transformation = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(flip_prob = 0.5),
        dense_transforms.ColorJitter(brightness = (0.7), contrast = (0.8), saturation = (0.7), hue = (0.2)),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])

    valid_transformation = dense_transforms.Compose([
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])

    # Initialize focal loss
    focal_loss_function = FocalLoss(alpha = 0.25, gamma = 2.0, reduction = 'mean').to(device)
    
    # Initialize size loss
    size_loss_function = torch.nn.MSELoss().to(device)

    # Load in data
    train_data = load_detection_data('dense_data/train', transform = train_transformation, batch_size = args.batch_size)
    valid_data = load_detection_data('dense_data/valid', transform = valid_transformation, batch_size = args.batch_size)

    # Initialize tb logging
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Initialize loss and AP trackers
    current_loss = float('inf')
    global_step = 0
    best_avg_ap = 0.0
    average_ap = 0.0

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

        # Print epoch and loss
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Calculate AP values for every 3rd epoch and print
        if (epoch) % 3 == 0:
            # Call evaluator
            model_evaluator = ModelDetectionEval(model, device)
            # Evaluate on validation data
            model_evaluator.evaluate(valid_data)
            # Get AP scores, also calculate overall AP (average)
            ap_scores = model_evaluator.calculate_ap_scores()
            average_ap = calculate_overall_ap(ap_scores)
            # Output stats
            print(f"Epoch {epoch+1} Evaluation: ")
            for category, scores in ap_scores.items():
                print(f"{category}: {scores}")
            print(f"Epoch {epoch+1} has average AP of {average_ap:.5f}")

        # Save model (based on overall average AP)
        if average_ap > best_avg_ap:
            best_avg_ap = average_ap
            save_model(model)
            print(f"Saving model at epoch {epoch+1} with average AP of {average_ap:.5f}")

        # Save model (based on loss)
        #if float(loss.item()) < current_loss:
            #current_loss = float(loss.item())
            #save_model(model)
            #print(f"Saving model at epoch {epoch+1} with loss of {loss.item()}")


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
