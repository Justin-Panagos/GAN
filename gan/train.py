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
from gan.utils import latent_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

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
    num_epochs = 151
    dataloader = get_data_loader()  # Use your dataset here

    # Loading the generator and discriminator models wiht the last checkpoint to continue training
    if os.path.exists("gan/models/gan_checkpoint_150_generator.pth"):
        load_checkpoint(
            generator, optimizer_G, "gan/models/gan_checkpoint_150_generator.pth"
        )
        load_checkpoint(
            discriminator,
            optimizer_D,
            "gan/models/gan_checkpoint_150_discriminator.pth",
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train the Discriminator
            optimizer_D.zero_grad()
            output_real = discriminator(
                real_images
            )  # Real images pass through discriminator
            loss_real = criterion(output_real, real_labels)  # Loss for real images
            loss_real.backward()  # Backpropagate the loss

            # Fake images (from generator)
            noise = torch.randn(batch_size, latent_vector).to(
                device
            )  # Generate random noise for generator input
            fake_images = generator(noise)  # Generate fake images
            output_fake = discriminator(
                fake_images.detach()
            )  # Discriminator sees fake images
            loss_fake = criterion(output_fake, fake_labels)  # Loss for fake images
            loss_fake.backward()  # Backpropagate the loss

            # Total discriminator loss
            optimizer_D.step()  # Update the discriminator's weights

            # Train the Generator
            optimizer_G.zero_grad()  # Zero the gradients for generator
            output_fake = discriminator(
                fake_images
            )  # Discriminator sees fake images (for generator update)
            loss_G = criterion(
                output_fake, real_labels
            )  # Loss for generator (wants to fool discriminator)
            loss_G.backward()  # Backpropagate the loss

            optimizer_G.step()  # Update the generator's weights

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {loss_real.item() + loss_fake.item()}, G Loss: {loss_G.item()}"
                )

        # Save generated images periodically
        if epoch % 10 == 0:
            save_checkpoint(
                generator,
                optimizer_G,
                epoch,
                filename=f"gan/models/gan_checkpoint_{epoch}_generator.pth",
            )
            save_checkpoint(
                discriminator,
                optimizer_D,
                epoch,
                filename=f"gan/models/gan_checkpoint_{epoch}_discriminator.pth",
            )


# Call train function
if __name__ == "__main__":
    train_gan()
