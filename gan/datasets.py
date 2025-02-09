# gan/datasets.py

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = self._get_all_image_files(data_dir)

    def _get_all_image_files(self, dir):
        image_files = []
        # Walk through subdirectories and collect image files
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):  # Filter for image files
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


# Define any image transformations (resize, normalize, etc.)
transform = transforms.Compose(
    [
        transforms.Resize(64),  # Resize to 64x64
        transforms.CenterCrop(64),  # Ensure images are square
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]
)


# Instantiate the dataset and DataLoader
def get_data_loader(batch_size=64):
    dataset = ImageDataset(data_dir="./datasets", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
