from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import torch.optim as optim

def train(args):
    from os import path

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Model
    model = Planner().to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    loss_function = torch.nn.MSELoss().to(device)

    # Set up data transformation
    transformation = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness = (0.8), contrast = (0.8), saturation = (0.8), hue = (0.2)),
        dense_transforms.ToTensor()
    ])

    # Load in data
    training_data = load_data('drive_data', transform = transformation, batch_size = args.batch_size)

    # Initialize loss
    current_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):

        # Set model to train
        model.train()

        # For each image/label
        for images, labels in training_data:

            # Send to device
            images = images.to(device)
            labels = labels.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Get predictions
            predictions = model(images)

            # Calculate loss
            loss = loss_function(predictions, labels)

            # Backward pass
            loss.backward()

            # Step in optimizer
            optimizer.zero_grad()

        
        # Validate
        model.eval()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Save model
        if float(loss.item()) < current_loss:
            current_loss = float(loss.item())
            save_model(model)
            print(f"Saving model at epoch {epoch + 1} with loss of {loss.item()}")


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--epochs', type=int, default=10) # Number of epochs
    parser.add_argument('--batch_size', type=int, default=32) # Batch size
    parser.add_argument('--lr', type=float, default=0.001) # Learning rate
    parser.add_argument('--wd', type=float, default=1e-4) # Weight decay
    args = parser.parse_args()
    train(args)
