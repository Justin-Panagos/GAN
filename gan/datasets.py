# gan/datasets.py

import os

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress warnings


class ImageLabelDataset(Dataset):
    def __init__(self, tfds_name="cats_vs_dogs", split="train", transform=None):
        """
        Args:
            tfds_name (str): Name of the TFDS dataset (e.g., "cats_vs_dogs").
            split (str): Dataset split (e.g., "train", "test").
            transform (callable, optional): Optional PyTorch transformations to apply to each image.
        """
        # Load the dataset from TFDS, including metadata
        dataset, info = tfds.load(
            tfds_name, with_info=True, as_supervised=True, split=split
        )

        def preprocess(image, label):
            # Resize to 192x192 (matching your GAN's target resolution)
            image = tf.image.resize(image, [128, 128])
            # Normalize to [-1, 1] for compatibility with your GAN
            image = (tf.cast(image, tf.float32) / 127.5) - 1.0
            # Return (image, label) for all samples (no filtering)
            return image, label

        # Apply preprocessing, filter out non-cat images, and prefetch for efficiency
        self.dataset = (
            dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(lambda image, label: image is not None)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.transform = transform  # Store PyTorch transformations if provided
        self.iterator = iter(self.dataset)  # Create an iterator for TensorFlow dataset

    def __len__(self):
        # Approximate length (TFDS doesn’t provide exact count easily, use info if needed)
        # cats_vs_dogs has ~12,500 cat images in the train split
        return 25000  # Approximate number of cat images

    def __getitem__(self, idx):
        # Fetch the next batch or retry if exhausted
        try:
            image, label = next(self.iterator)
            # Convert TensorFlow tensor to NumPy, then to PyTorch tensor
            image = tf.keras.preprocessing.image.img_to_array(image.numpy())
            image = (
                torch.from_numpy(image).permute(2, 0, 1).float()
            )  # [H, W, C] -> [C, H, W]
            # Apply PyTorch transformations if specified (e.g., additional augmentation)
            if self.transform:
                image = self.transform(image)
            return image, label.numpy()  # Return image tensor and label (0 for cats)
        except StopIteration:
            self.iterator = iter(self.dataset)  # Reset iterator if exhausted
            return self.__getitem__(idx)


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally to add variety
        transforms.RandomRotation(
            10
        ),  # Randomly rotate images by up to 10 degrees for augmentation
    ]
)


# Function to create a DataLoader for batching and shuffling the dataset
def get_data_loader(batch_size=8, tfds_name="cats_vs_dogs", split="train"):
    """
    Create a PyTorch DataLoader for the specified TFDS dataset.

    Args:
        batch_size (int): Batch size for training (default 8 for stability).
        tfds_name (str): Name of the TFDS dataset (e.g., "cats_vs_dogs").
        split (str): Dataset split (e.g., "train", "test").
    """
    # Instantiate the dataset with TFDS and transformations
    dataset = ImageLabelDataset(tfds_name=tfds_name, split=split, transform=transform)
    # Create a DataLoader to batch and shuffle the data for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
