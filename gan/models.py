# gan/models.py

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.functional as F

from gan.utils import latent_vector


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Increase number of filters/channels and add more layers
        self.model = nn.Sequential(
            # Input is a latent vector z (latent_vector = 512)
            nn.ConvTranspose2d(
                latent_vector, 1024, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # First layer with more channels and kernel size 4
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Second layer with increased feature map size
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Third layer, larger feature maps and kernels
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Fourth layer with higher resolution output
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Final layer that outputs an image (output size 64x64 for instance)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Tanh to scale the output to [-1, 1]
        )

    def forward(self, z):
        # Reshape the latent vector z into a 4D tensor [batch_size, 512, 1, 1]
        z = z.view(
            z.size(0), 512, 1, 1
        )  # Reshaping to make it compatible with ConvTranspose2d layers
        return self.model(z)


# Define the Discriminator model (DCGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # More powerful discriminator with more layers and larger channels
        self.model = nn.Sequential(
            # Input is a 64x64 image (or whatever size you are generating)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Second layer with more filters
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Third layer
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Fourth layer
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final layer to output a single probability value (real or fake)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.model(x)
