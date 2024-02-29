import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim
import torch.nn.functional as F


def adjust_class_weights(per_class_accuracy, base_weights):
    # Adjust class weights based on their accuracy
    # Calculate inverse accuracy
    inverse_accuracy = 1.0 - per_class_accuracy
    # Normalize weights
    normalized_weights = (inverse_accuracy / inverse_accuracy.sum()) * base_weights.sum()
    return normalized_weights

def calculate_per_class_accuracy(confusion_matrix):
    # Calculate class accuracy
    correct_in_class = confusion_matrix.diag()
    total_in_class = confusion_matrix.sum()
    per_class_accuracy = correct_in_class / total_in_class
    return per_class_accuracy.nan_to_num(0)

def train(args):

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    from os import path
    
    # Get device
    device = torch.device(args.device)

    # Initialize model
    model = FCN().to(device)
    
    # Create loss function and use dense_class_distribution
    base_weights = torch.tensor([1.0 / i for i in DENSE_CLASS_DISTRIBUTION], dtype = torch.float32).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight = base_weights)

    # Create optimizer, use model parameters and learning rate
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    print(f'Learning rate: {args.lr}')
    print(f'Device: {args.device}')

    # Load in data
    training_data = load_dense_data('dense_data/train')
    validation_data = load_dense_data('dense_data/valid')

    # Initialize data augmentations
    augmentations = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness = (0.5), contrast = (0.5), saturation = (0.5), hue = (0.5)),
    ])

    # Initialize loggers for tensorboard
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Keep track of best IoU value
    best_val_iou = float('-inf')

    # Initialize global step for tensorboard logging
    global_step = 0

    # Training loop
    for epoch in range(args.epochs):

        # Set model to train
        model.train()

        # Run training
        for inputs, labels in training_data:

            # Augment inputs
            inputs_augmented, labels_augmented = augmentations(inputs, labels)

            # Send to device
            inputs_augmented, labels_augmented = inputs_augmented.to(device), labels_augmented.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_augmented)

            # Calculate loss
            loss = loss_function(outputs, labels_augmented.long())

            # Backward pass
            loss.backward()

            # Step in optimizer
            optimizer.step()

            # Log training loss in tb
            if train_logger:
                train_logger.add_scalar('loss', loss.item(), global_step = global_step)

            # Increment global step
            global_step += 1

        # Validate
        model.eval()
        validation_iou, validation_loss, correct, total = 0, 0, 0, 0
        confusion_matrix = ConfusionMatrix(size = 5)
        # Run through
        with torch.no_grad():
            for inputs, labels in validation_data:
                # Send to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Get outputs
                outputs = model(inputs)
                # Calculate loss
                loss = loss_function(outputs, labels.long())
                validation_loss += loss.item()
                # Update confusion matrix
                preds = torch.argmax(outputs, dim = 1)
                confusion_matrix.add(preds, labels)
                # Calculate accuracy
                correct += (preds == labels).sum().item()
                total += labels.numel()

            # Calculate stats
            validation_iou = confusion_matrix.iou.item()
            validation_loss /= len(validation_data)
            validation_accuracy = correct / total

        # Log stats
        if valid_logger:
            valid_logger.add_scalar('iou', validation_iou, global_step = global_step)
            valid_logger.add_scalar('loss', validation_loss, global_step = global_step)
            valid_logger.add_scalar('accuracy', validation_accuracy, global_step = global_step)

        # Print out stats
        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {loss.item()}, '
              f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}, '
              f'Validation IoU: {validation_iou}')

        # Calculate per-class stats
        per_class_accuracy = calculate_per_class_accuracy(confusion_matrix)
        adjusted_weights = adjust_class_weights(per_class_accuracy, base_weights)
        loss_function = torch.nn.CrossEntropyLoss(weight = adjusted_weights.to(device))
        print(f'Epoch: {epoch+1}, Adjusted Weights: {adjusted_weights}, Per-Class Accuracy: {per_class_accuracy}')
        
        # Check if current IoU is best so far
        if validation_iou > best_val_iou:
            best_val_iou = validation_iou
            print(f'Saving model at epoch {epoch+1} with Validation Accuracy: {validation_accuracy}, '
                  f'and Validation IoU: {validation_iou}')
            save_model(model)





def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--epochs', type = int, default = 25) # Number of epochs
    parser.add_argument('--lr', type = float, default = 1e-4) # Learning rate
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu') # Default device

    args = parser.parse_args()
    train(args)
