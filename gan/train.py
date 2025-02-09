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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def save_checkpoint(model, optimizer, epoch, filename="gan_checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


# Training loop
def train_gan():
    num_epochs = 50
    dataloader = get_data_loader()  # Use your dataset here

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train the Discriminator
            optimizer_D.zero_grad()
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward()

            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward()

            optimizer_D.step()

            # Train the Generator
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_images)
            loss_G = criterion(output_fake, real_labels)
            loss_G.backward()

            optimizer_G.step()

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
