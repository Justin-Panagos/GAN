# gan/utils.py

from torchvision.utils import save_image


def save_generated_images(images, epoch):
    images = images / 2 + 0.5  # Rescale to [0, 1]
    save_image(images, f"generated_images/epoch_{epoch}.png")
