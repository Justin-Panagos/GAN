# gan/utils.py

from torchvision.utils import save_image

# setting global varibles to be ued in
latent_vector = 512


def save_generated_images(images, epoch):
    images = images / 2 + 0.5  # Rescale to [0, 1]
    save_image(images, f"epoch_{epoch}.png")
