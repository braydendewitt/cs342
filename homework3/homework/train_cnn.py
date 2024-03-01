from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.optim as optim
import torchvision.transforms as transforms


def train(args):
    from os import path
    
    # Initialize model and send to GPU/device
    model = CNNClassifier().to(args.device)

    # Create loss function using cross-entropy
    loss_function = torch.nn.CrossEntropyLoss()

    # Create optimizer, use model parameters and learning rate
    #optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = 1e-3)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma = 0.10)
    print(f'Learning rate: {args.lr}')
    print(f'Device: {args.device}')

    # Load in data
    training_data = load_data('data/train')
    validation_data = load_data('data/valid')

    # Initialize data augementations
    tensor_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
        transforms.RandomRotation(10, fill = (0,)),
        transforms.RandomPerspective(),
    ])

    # Initialize loggers for tensorboard
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Keep track of best validation accuracy
    best_val_accuracy = float('-inf')

    # Initialize global step (for tb logging)
    global_step = 0

    # Training loop
    for epoch in range(args.epochs):
        # Set model to train
        model.train()
        # Run training
        for inputs, labels in training_data:
            # Augment data and send to device
            inputs_augmented = tensor_transformations(inputs)
            inputs_augmented, labels = inputs_augmented.to(args.device), labels.to(args.device)
            # Zero gradient
            optimizer.zero_grad()
            # Forward pass to get predicted outputs
            outputs = model(inputs_augmented)
            # Calculate loss
            loss = loss_function(outputs, labels)
            # Backward pass to calculate gradient
            loss.backward()
            # Step in optimizer
            optimizer.step()

            # Log training loss in tb
            if train_logger:
                train_logger.add_scalar('loss', loss.item(), global_step=global_step)

            # Increment gloabl step
            global_step += 1
    
        # Validate
        # Set model to eval mode
        model.eval()
        validation_loss = 0
        validation_accuracy = 0
        # Run through
        with torch.no_grad():
            for inputs, labels in validation_data:
                # Send to device
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                # Get outputs
                outputs = model(inputs)
                # Calculate loss
                validation_loss_item = loss_function(outputs, labels).item()
                validation_loss += validation_loss_item
                # Calculate accuracy
                validation_accuracy += accuracy(outputs, labels).item()
        # Calculate average validation loss and accuracy
        validation_accuracy /= len(validation_data)
        validation_loss /= len(validation_data)

        # Log validation accuracy in tb
        if valid_logger:
            valid_logger.add_scalar('accuracy', validation_accuracy, global_step=global_step)
    
        # Print out stats
        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {loss.item()}, '
              f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')
    
        # Check if current validation accuracy is the best so far
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            print(f'Saving model at epoch {epoch+1} with Validation Accuracy: {validation_accuracy}')
            save_model(model)
        
        # Step in LR scheduler
            scheduler.step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir') # tb log directory
    parser.add_argument('--epochs', type = int, default = 25) # Number of epochs
    parser.add_argument('--lr', type = float, default = 1e-4) # Learning rate
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu') # Default device
    
    args = parser.parse_args()
    train(args)
