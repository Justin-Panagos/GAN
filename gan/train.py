# gan/train.py

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs("datasets/models", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from gan.datasets import get_data_loader  # Use your dataset here
from gan.models import Discriminator, Generator
from gan.utils import clip_weights, latent_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Binary Cross Entropy Loss, used for a regular GAN
# criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint {filename}, starting from epoch {epoch}")
    return epoch


def save_checkpoint(model, optimizer, epoch, filename="gan_checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


# Training loop2
def train_gan():
    num_epochs = 101
    dataloader = get_data_loader()  # Use your dataset here
    n_critic = 5

    # Loading the generator and discriminator models wiht the last checkpoint to continue training
    if os.path.exists("datasets/models/gan_checkpoint_100_generator.pth"):
        load_checkpoint(
            generator, optimizer_G, "datasets/models/gan_checkpoint_100_generator.pth"
        )
        load_checkpoint(
            discriminator,
            optimizer_D,
            "datasets/models/gan_checkpoint_100_discriminator.pth",
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(n_critic):
                # Train the Discriminator
                optimizer_D.zero_grad()
                # Real images pass through discriminator
                output_real = discriminator(real_images)
                loss_real = output_real.mean()

                # Fake images (from generator)
                # Generate random noise for generator input
                noise = torch.randn(batch_size, latent_vector).to(device)
                # Generate fake images
                fake_images = generator(noise)
                # Discriminator sees fake images
                output_fake = discriminator(fake_images.detach())
                loss_fake = output_fake.mean()  # Loss for fake images

                loss_D = loss_fake - loss_real
                loss_D.backward()  # Backpropagate the loss

                # Total discriminator loss
                optimizer_D.step()  # Update the discriminator's weights

                clip_weights(discriminator)

            # Train the Generator
            optimizer_G.zero_grad()  # Zero the gradients for generator
            # Discriminator sees fake images (for generator update)
            output_fake = discriminator(fake_images)
            loss_G = (
                -output_fake.mean()
            )  # Loss for generator (wants to fool discriminator)
            loss_G.backward()  # Backpropagate the loss

            optimizer_G.step()  # Update the generator's weights

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {loss_D.item()}, G Loss: {loss_G.item()}"
                )

        # Save generated images periodically
        if epoch % 20 == 0:
            save_checkpoint(
                generator,
                optimizer_G,
                epoch,
                filename=f"datasets/models/gan_checkpoint_{epoch}_generator.pth",
            )
            save_checkpoint(
                discriminator,
                optimizer_D,
                epoch,
                filename=f"datasets/models/gan_checkpoint_{epoch}_discriminator.pth",
            )


# Call train function
if __name__ == "__main__":
    train_gan()
