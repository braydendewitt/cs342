from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import csv
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        
        # Create lists to store paths and labels
        self.data = []
        self.labels = []

        # Create label names
        self.label_names = LABEL_NAMES

        # Ensure correct image size (64x64) and store as tensor
        self.transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])

        # Read in CSV file of labels
        labels_path = os.path.join(dataset_path, 'labels.csv') # Get labels datapath
        with open(labels_path, 'r') as csvfile: # Open the file
            reader = csv.DictReader(csvfile) # Create reader
            for row in reader:
                # Get image path
                image_path = os.path.join(dataset_path, row['file'])
                # Get label
                label = self.label_names.index(row['label'])
                # Add path and label to list
                self.data.append(image_path)
                self.labels.append(label)

    def __len__(self):

        # Get length of dataset
        return len(self.data)

    def __getitem__(self, idx):
        
        # Read in image
        image = Image.open(self.data[idx]).convert('RGB')

        # Transform image to tensor
        image = self.transform(image)

        # Get label
        label = self.labels[idx]

        # Return tuple
        return image, label

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
