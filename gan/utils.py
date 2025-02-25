# gan/utils.py

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# setting global varibles to be ued in
latent_vector = 512


# Function to save generated images, rescaling from [-1, 1] to [0, 1] for visualization
def save_generated_images(images, epoch):
    # Rescale images from [-1, 1] to [0, 1] for saving
    images = images / 2 + 0.5
    # Save the images to a file in the generated_images directory
    save_image(images, f"datasets/generated_images/epoch_{epoch}.png")


# Function to clip the weights of the discriminator (critic)
def compute_gradient_penalty(D, real_samples, fake_samples, labels, device="cuda"):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    # Get critic score for interpolated samples, conditioned on labels
    d_interpolates = D(interpolates, labels)

    # Create a tensor of ones for gradient computation
    fake = torch.ones(d_interpolates.size()).to(device)

    # Compute gradients with respect to interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Reshape gradients to compute norms
    gradients = gradients.view(gradients.size(0), -1)

    # Calculate gradient penalty as the mean squared difference from norm=1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
