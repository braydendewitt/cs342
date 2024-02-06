from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def train(args):
    # Initialize model
    model = model_factory[args.model]().to(device)

    # Create loss function
    loss_function = ClassificationLoss()

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), args.lr)
    print(args.lr)

    # Load in data
    training_data = load_data('data/train')
    validation_data = load_data('data/valid')

    #################### CHECK IF LOADED CORRECTLY #############

    best_val_accuracy = float('-inf')

    # Training loop
    for epoch in range(args.epochs):
        # Set model to train
        model.train()
        # Run SGD
        for inputs, labels in training_data:
            # send to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradient
            optimizer.zero_grad()
            # Get outputs
            outputs = model(inputs)
            # Get loss
            loss = loss_function(outputs, labels)
            # Backward step
            loss.backward()
            # Step in optimizer
            optimizer.step()
        
        # Validate
        model.eval()
        validation_loss = 0
        validation_accuracy = 0
        # Run through
        with torch.no_grad():
            for inputs, labels in validation_data:
                # send to gpu
                inputs, labels = inputs.to(device), labels.to(device)
                # Get outputs
                outputs = model(inputs)
                # Get loss and accuracy
                validation_loss += loss_function(outputs, labels).item()
                validation_accuracy += accuracy(outputs, labels).item()
        # Calculate loss and accuracy
        validation_accuracy /= len(validation_data)
        validation_loss /= len(validation_data)

        # Print out stats
        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {loss.item()}, '
              f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')

        # Check if current validation accuracy is the best so far
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            print(f'Saving model at epoch {epoch+1} with Validation Accuracy: {validation_accuracy}')
            save_model(model)

    # Save model
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('--epochs', type = int, default = 25)
    parser.add_argument('--lr', type=float, default = 1e-4)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
