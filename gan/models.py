# gan/models.py

import torch
import torch.nn as nn
import torch.nn.functional as func

from gan.utils import latent_vector


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Input noise vector of size 100 (latent vector)
        self.fc1 = nn.Linear(
            latent_vector, 256 * 4 * 4
        )  # Start from a 4x4 image (or smaller size)

        # Transposed convolutions to upsample to 64x64x3 image
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # Upsample to 8x8
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # Upsample to 16x16
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # Upsample to 32x32
        self.deconv4 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # Upsample to 64x64

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc1(z)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape to 4x4x256

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.tanh(
            self.deconv4(x)
        )  # Output image with values in [-1, 1] (due to Tanh activation)

        return x


# Define the Discriminator model (DCGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Convolutional layers to downsample the image
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=4, stride=2, padding=1
        )  # Downsample to 32x32
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # Downsample to 16x16
        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # Downsample to 8x8
        self.conv4 = nn.Conv2d(
            256, 512, kernel_size=4, stride=2, padding=1
        )  # Downsample to 4x4

        # Output layer (1 unit for binary classification)
        self.fc1 = nn.Linear(512 * 4 * 4, 1)

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = self.sigmoid(x)  # Output a probability of the image being real or fake

        return x
