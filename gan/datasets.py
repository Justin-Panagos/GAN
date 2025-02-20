# gan/datasets.py

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Define a custom dataset class to load images from a directory
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory containing images.
            transform (callable, optional): Transformations to apply to each image.
        """
        self.data_dir = data_dir  # Store the directory path
        self.transform = transform  # Store the transformation pipeline
        self.image_files = self._get_all_image_files(
            data_dir
        )  # Collect all image file paths

    def _get_all_image_files(self, dir):
        # Helper function to recursively find all image files in subdirectories
        image_files = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                # Filter for common image extensions (case-insensitive)
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess the image at the given index
        img_name = self.image_files[idx]  # Get the file path
        image = Image.open(img_name).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        return image


# Define the transformation pipeline for preprocessing images
transform = transforms.Compose(
    [
        transforms.Resize(192),  # Resize images to 64x64 pixels
        transforms.CenterCrop(192),  # Crop to ensure a square 64x64 output
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally to add variety
        transforms.RandomRotation(
            10
        ),  # Randomly rotate images by up to 10 degrees for augmentation
        transforms.ToTensor(),  # Convert image to a PyTorch tensor (0-1 range)
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize to [-1, 1] range
    ]
)


# Function to create a DataLoader for batching and shuffling the dataset
def get_data_loader(batch_size=16, data_dir="./datasets/images"):
    # Instantiate the dataset with the specified directory and transformations
    dataset = ImageDataset(data_dir=data_dir, transform=transform)
    # Create a DataLoader to batch and shuffle the data for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
