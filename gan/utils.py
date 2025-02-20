# gan/utils.py

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# setting global varibles to be ued in
latent_vector = 512


def save_generated_images(images, epoch):
    images = images / 2 + 0.5  # Rescale to [0, 1]
    save_image(images, f"epoch_{epoch}.png")


# Function to clip the weights of the discriminator (critic)
def compute_gradient_penalty(D, real_samples, fake_samples, device="cuda"):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
