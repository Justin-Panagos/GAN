# gan/utils.py

import glob

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


# Function to load a model from the latest checkpoint
def load_model_from_checkpoint(
    model, optimizer=None, model_type="generator", device=None
):
    try:
        # Find all the checkpoint files for the model type (generator or discriminator)
        checkpoint_files = glob.glob(f"datasets/models/*{model_type}.pth")
        if not checkpoint_files:
            print(f"No {model_type} checkpoint found!")
            return None  # Return None if no checkpoint is found

        # Sort files by epoch number (latest first)
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        # Load the checkpoint file
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)  # Default to 0 if epoch not in checkpoint
        print(
            f"Loaded {model_type} model and optimizer from {latest_checkpoint}, starting from epoch {epoch}"
        )
        return epoch  # Return epoch to resume training

    except Exception as e:
        print(f"Error loading {model_type} model: {str(e)}")
        return None


# Function to clip the weights of the discriminator (critic)
def compute_gradient_penalty(D, real_samples, fake_samples, labels, device="cpu"):
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


# Function to load a saved checkpoint for resuming training
# Load the checkpoint file (contains model weights and optimizer state)
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint {filename}, starting from epoch {epoch}")
    return epoch
