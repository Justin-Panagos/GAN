# gan/train.py

import os
import sys

import torch
import torch.optim as optim

# Create the directory to save model checkpoints if it doesn’t exist
os.makedirs("datasets/models", exist_ok=True)
os.makedirs("datasets/progression_check", exist_ok=True)
# Add the parent directory to the system path so we can import from gan/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules from your gan package
from gan.datasets import get_data_loader  # Loads the dataset of real images
from gan.models import Discriminator, Generator  # The WGAN-GP models
from gan.utils import (
    compute_gradient_penalty,  # Utility functions
    latent_vector,
    load_model_from_checkpoint,
    save_checkpoint,
)

# Set the device to GPU if available, otherwise CPU
""" My GPU was creating a bottle neck for some time, overuse maybe ?? cant flush cache?? not sure """
# "cuda" if torch.cuda.is_available() else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator and discriminator models, move them to the device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers for both models with WGAN-GP-friendly hyperparameters
# - lr=0.0002 for generator, lr=0.0001 for discriminator (slower learning for critic)
# - betas=(0, 0.9) are standard for WGAN-GP stability
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00003, betas=(0.0, 0.9))


# Main training function for the WGAN-GP
def train_gan():
    num_epochs = 10  # Train for 201 epochs since dataset is small (100 images)
    dataloader = get_data_loader(batch_size=16, tfds_name="cats_vs_dogs", split="train")
    n_critic = 5  # Train critic 5 times per generator step (WGAN-GP standard)
    lambda_gp = 10  # Weight for gradient penalty term in critic loss

    start_epoch_g = load_model_from_checkpoint(
        generator, optimizer_G, "generator", device
    )
    if start_epoch_g is None:
        print("No generator checkpoint found, starting from scratch.")

    # Load discriminator checkpoint (if any)
    start_epoch_d = load_model_from_checkpoint(
        discriminator, optimizer_D, "discriminator", device
    )
    if start_epoch_d is None:
        print("No discriminator checkpoint found, starting from scratch.")

    # Training loop over epochs
    for epoch in range(num_epochs):
        # Iterate over batches of real images
        for i, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)  # [batch_size, 3, 192, 192]
            labels = labels.to(device).long()
            batch_size = real_images.size(0)

            noise = torch.randn_like(real_images) * 0.05
            real_images = real_images + noise

            # Train the critic (discriminator) more often than the generator
            # with torch.cuda.amp.autocast():
            for _ in range(n_critic):
                optimizer_D.zero_grad()  # Clear previous gradients

                # Score real images with the critic
                output_real = discriminator(real_images, labels)
                loss_real = output_real.mean()  # Average score for real images

                # Generate fake images from random noise
                noise = torch.randn(batch_size, latent_vector, 1, 1).to(device)
                fake_images = generator(noise, labels)
                # Score fake images with the critic (detach to avoid generator gradients)
                output_fake = discriminator(fake_images.detach(), labels)
                loss_fake = output_fake.mean()  # Average score for fake images

                # Compute gradient penalty to enforce Lipschitz constraint
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_images, fake_images, labels, device=device
                )

                # Total critic loss: Wasserstein distance + gradient penalty
                loss_D = (loss_fake - loss_real) + lambda_gp * gradient_penalty
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()  # Clear previous gradients
            # Re-score fake images (with gradients this time)
            output_fake = discriminator(fake_images, labels)
            # Generator loss: maximize critic’s score on fakes
            loss_G = -output_fake.mean()
            loss_G.backward(retain_graph=True)  # Backpropagate generator loss
            optimizer_G.step()  # Update generator weights

            # Print progress every 10 steps (more frequent with small dataset)
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}],",
                    f"D Loss: {round(loss_D.item(),2)}, G Loss: {round(loss_G.item(),2)}",
                    f"Real: {round(loss_real.item(),2)}, Fake: {round(loss_fake.item(),2)}, GP: {round(gradient_penalty.item(),5)}",
                )

        # Save model checkpoints every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                generator,
                optimizer_G,
                epoch,
                f"datasets/models/chp_{epoch}_generator.pth",
            )
            save_checkpoint(
                discriminator,
                optimizer_D,
                epoch,
                f"datasets/models/chp_{epoch}_discriminator.pth",
            )


# Run the training function if this file is executed directly
if __name__ == "__main__":
    train_gan()


# Step [1620/3125], [1630/3125], [1640/3125], [1650/3125]: These are the steps within Epoch 0,
# showing progress through the dataset. Each step processes one batch (8 images).

# D Loss: The critic (discriminator) loss in the WGAN-GP,
# calculated as (loss_fake - loss_real) + lambda_gp * gradient_penalty.
# It should ideally stabilize near 0 as training progresses, indicating a balanced critic.

# G Loss: The generator loss, calculated as -output_fake.mean().
# It represents how well the generator fools the critic,
# and it should stabilize (not swing wildly) as the generator improves.

# Real: The critic’s average score for real images (output_real.mean()). In WGAN-GP,
# this should be positive, ideally around 1 or higher, as the critic assigns high scores to real data.

# Fake: The critic’s average score for fake (generated) images (output_fake.mean()).
# This should be negative, ideally around -1 or lower, as the critic assigns low scores to fake data.

# GP (Gradient Penalty): The gradient penalty term (lambda_gp * gradient_penalty) ensures the
# critic satisfies the Lipschitz constraint. It should be small (e.g., ~0.1–1) and stable,
# indicating the critic isn’t overfitting or exploding.


"""Extra items of code to be used when during/after debugging steps"""

## add this if statment in into teh train_gan() funciton before starting on the for loop for each epoch if wanting ot train from a certain checkpoint
# Check for existing checkpoints to resume training; otherwise start fresh
# if os.path.exists("datasets/models/chp_65_generator.pth"):
#     load_checkpoint(
#         generator, optimizer_G, "datasets/models/chp_65_generator.pth   "
#     )
#     load_checkpoint(
#         discriminator, optimizer_D, "datasets/models/chp_65_discriminator.pth"
#     )
# else:
#     print("No checkpoint found, starting training from scratch.")
