# gan/utils.py

from torchvision.utils import save_image

# setting global varibles to be ued in
latent_vector = 512


def save_generated_images(images, epoch):
    images = images / 2 + 0.5  # Rescale to [0, 1]
    save_image(images, f"epoch_{epoch}.png")


# Function to clip the weights of the discriminator (critic)
def clip_weights(model, clip_value=0.011):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
