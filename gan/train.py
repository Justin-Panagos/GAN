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
from gan.utils import compute_gradient_penalty  # Utility functions
from gan.utils import latent_vector, save_generated_images

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator and discriminator models, move them to the device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers for both models with WGAN-GP-friendly hyperparameters
# - lr=0.0002 for generator, lr=0.0001 for discriminator (slower learning for critic)
# - betas=(0, 0.9) are standard for WGAN-GP stability
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.0, 0.9))


# Function to load a saved checkpoint for resuming training
# Load the checkpoint file (contains model weights and optimizer state)
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint {filename}, starting from epoch {epoch}")
    return epoch


# Function to save the current state of a model and optimizer
# Save the epoch number, model weights, and optimizer state to a file
def save_checkpoint(model, optimizer, epoch, filename="gan_checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


# Main training function for the WGAN-GP
def train_gan():
    num_epochs = 61  # Train for 201 epochs since dataset is small (100 images)
    dataloader = get_data_loader(
        batch_size=16
    )  # Load data with batch size 16 (~6 batches/epoch)
    n_critic = 1  # Train critic 5 times per generator step (WGAN-GP standard)
    lambda_gp = 20  # Weight for gradient penalty term in critic loss

    # Check for existing checkpoints to resume training; otherwise start fresh
    if os.path.exists("datasets/models/chp_60_generator.pth"):
        load_checkpoint(generator, optimizer_G, "datasets/models/chp_60_generator.pth")
        load_checkpoint(
            discriminator, optimizer_D, "datasets/models/chp_60_discriminator.pth"
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training loop over epochs
    for epoch in range(num_epochs):
        # Iterate over batches of real images
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)  # Move batch to GPU/CPU
            # Get the current batch size (might be <16 for last batch)
            batch_size = real_images.size(0)

            # Train the critic (discriminator) more often than the generator
            for _ in range(n_critic):
                optimizer_D.zero_grad()  # Clear previous gradients

                # Score real images with the critic
                output_real = discriminator(real_images)
                loss_real = output_real.mean()  # Average score for real images

                # Generate fake images from random noise
                noise = torch.randn(batch_size, latent_vector, 1, 1).to(device)
                fake_images = generator(noise)
                # Score fake images with the critic (detach to avoid generator gradients)
                output_fake = discriminator(fake_images.detach())
                loss_fake = output_fake.mean()  # Average score for fake images

                # Compute gradient penalty to enforce Lipschitz constraint
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_images, fake_images, device=device
                )

                # Total critic loss: Wasserstein distance + gradient penalty
                loss_D = (loss_fake - loss_real) + lambda_gp * gradient_penalty
                loss_D.backward(retain_graph=True)  # Backpropagate critic loss
                optimizer_D.step()  # Update critic weights

            # Train the generator
            optimizer_G.zero_grad()  # Clear previous gradients
            # Re-score fake images (with gradients this time)
            output_fake = discriminator(fake_images)
            # Generator loss: maximize critic’s score on fakes
            loss_G = -output_fake.mean()
            loss_G.backward(retain_graph=True)  # Backpropagate generator loss
            optimizer_G.step()  # Update generator weights

            # Print progress every 10 steps (more frequent with small dataset)
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {loss_D.item()}, G Loss: {loss_G.item()}"
                )

            # # Save generated images every 50 steps for debugging
            # if i % 50 == 0:
            #     with torch.no_grad():  # Disable gradients for efficiency
            #         fake_images = generator(noise)  # Generate fresh images
            #         save_generated_images(
            #             fake_images,
            #             f"datasets/progression_check/epoch_{epoch}_step_{i}",
            #         )

        # Save model checkpoints every 20 epochs
        if epoch % 10 == 0:
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
