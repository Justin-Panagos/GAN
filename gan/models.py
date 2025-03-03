# gan/models.py

import torch
import torch.nn as nn

from gan.utils import latent_vector


# Define the Generator class, now conditioning on labels (e.g., 0 for cats, 1 for dogs)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(
            2, 32
        )  # 2 classes (cat=0, dog=1), 32-dim embedding
        self.model = nn.Sequential(
            # 1x1 → 4x4
            nn.ConvTranspose2d(latent_vector + 32, 64, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 4x4 → 8x8
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(4, 3, 4, 2, 1, bias=False),  # 128x128, 3 channels (RGB)
            nn.Tanh(),  # Scale output to [-1, 1]
        )

    def forward(self, z, labels):
        label_emb = (
            self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        )  # [batch_size, 32, 1, 1]
        z = z.view(z.size(0), latent_vector, 1, 1)  # [batch_size, latent_vector, 1, 1]
        combined = torch.cat(
            [z, label_emb], dim=1
        )  # [batch_size, latent_vector+32, 1, 1]
        return self.model(combined)


# Define the Discriminator (Critic) class, now processing images and labels
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(2, 32)  # 2 classes, 32-dim embedding
        self.img_net = nn.Sequential(
            # 128x128 → 32x32
            nn.Conv2d(3, 32, 4, 4, 0, bias=False),  # 32x32 (128/4 = 32)
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 → 8x8
            nn.Conv2d(32, 64, 4, 4, 0, bias=False),  # 8x8 (32/4 = 8)
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final = nn.Conv2d(
            64 + 32, 1, 8, 1, 0, bias=False
        )  # +32 for label, 8x8 → 1x1

    def forward(self, x, labels):
        img_features = self.img_net(x)  # [batch_size, 64, 8, 8]
        label_emb = self.label_emb(labels).view(-1, 32, 1, 1)  # [batch_size, 32, 1, 1]
        label_emb = label_emb.repeat(1, 1, 8, 8)  # [batch_size, 32, 8, 8]
        combined = torch.cat(
            [img_features, label_emb], dim=1
        )  # [batch_size, 64+32, 8, 8]
        return self.final(combined)


#  still too complex for small datasets
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             # Layer 1: Start with latent vector, output 4x4
#             nn.ConvTranspose2d(latent_vector, 128, 4, 1, 0, bias=False),  # 4x4
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # Layer 2: Double size to 8x8
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 8x8
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # Layer 3: Double size to 16x16
#             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 16x16
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             # Layer 4: Triple size to 48x48 with output_padding tweak
#             nn.ConvTranspose2d(32, 16, 4, 3, 1, output_padding=1, bias=False),  # 48x48
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             # Layer 5: Quadruple size to 192x192
#             nn.ConvTranspose2d(16, 3, 6, 4, 1, bias=False),  # 192x192
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         z = z.view(z.size(0), latent_vector, 1, 1)
#         return self.model(z)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 4, 1, bias=False),  # 192/4 = 48x48
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 3, 1, bias=False),  # 48/3 = 16x16
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8x8
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 1, 8, 1, 0, bias=False),  # 1x1
#         )

#     def forward(self, x):
#         return self.model(x)


# # # these models need to be simpliefe further
# # # Define the Generator class, now generating 192x192x3 images (triple the original 64x64)
# # class Generator(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         # Define the sequential model to upsample from latent vector to 192x192x3
# #         self.model = nn.Sequential(
# #             # Step 1: Start with latent vector (512 channels), upsample to 256 channels
# #             # - 4x4 kernel, stride 1, no padding: outputs 4x4
# #             nn.ConvTranspose2d(latent_vector, 256, 4, 1, 0, bias=False),
# #             nn.BatchNorm2d(256),
# #             nn.ReLU(True),
# #             # Step 2: Upsample to 128 channels, double size to 8x8
# #             # - 4x4 kernel, stride 2, padding 1: outputs 8x8
# #             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(128),
# #             nn.ReLU(True),
# #             # Step 3: Upsample to 64 channels, double size to 16x16
# #             # - 4x4 kernel, stride 2, padding 1: outputs 16x16
# #             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(True),
# #             # Step 4: Upsample to 32 channels, double size to 32x32
# #             # - 4x4 kernel, stride 2, padding 1: outputs 32x32
# #             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(True),
# #             # Step 5: Upsample to 16 channels, double size to 64x64
# #             # - 4x4 kernel, stride 2, padding 1: outputs 64x64
# #             nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(16),
# #             nn.ReLU(True),
# #             # Step 6: Upsample to 3 channels (RGB), triple size to 192x192
# #             # - 6x6 kernel, stride 3, padding 1: outputs 192x192 (64*3=192)
# #             nn.ConvTranspose2d(16, 3, 6, 3, 1, bias=False),
# #             nn.Tanh(),  # Scale output to [-1, 1]
# #         )

# #     def forward(self, z):
# #         # Reshape latent vector (batch_size, 512) to 4D (batch_size, 512, 1, 1)
# #         z = z.view(z.size(0), latent_vector, 1, 1)
# #         # Generate a 192x192x3 image from the latent vector
# #         return self.model(z)


# # # Define the Discriminator (Critic) class, now processing 192x192x3 images
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         # Define the sequential model to downsample 192x192x3 to a single score
# #         self.model = nn.Sequential(
# #             # Step 1: Input 192x192x3, downsample to 64 channels
# #             # - 4x4 kernel, stride 3, padding 1: outputs 64x64 (192/3=64)
# #             nn.Conv2d(3, 64, 4, 3, 1, bias=False),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # Step 2: Downsample to 128 channels, halve size to 32x32
# #             # - 4x4 kernel, stride 2, padding 1: outputs 32x32
# #             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # Step 3: Downsample to 256 channels, halve size to 16x16
# #             # - 4x4 kernel, stride 2, padding 1: outputs 16x16
# #             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # Step 4: Downsample to 512 channels, halve size to 8x8
# #             # - 4x4 kernel, stride 2, padding 1: outputs 8x8
# #             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # Step 5: Final downsample to 1 channel (score)
# #             # - 8x8 kernel, stride 1, no padding: outputs 1x1 (fully collapses 8x8)
# #             nn.Conv2d(512, 1, 8, 1, 0, bias=False),
# #         )

# #     def forward(self, x):
# #         # Process a 192x192x3 image and output a single score
# #         return self.model(x)


# # # # these models will be good to use when im training on 10,000 + images
# # # class Generator(nn.Module):
# # #     """Current model setup is too complex for a small dataset to train on,"""
# # #     def __init__(self):
# # #         super(Generator, self).__init__()

# # #         # Increase number of filters/channels and add more layers
# # #         self.model = nn.Sequential(
# # #             # Input is a latent vector z (latent_vector = 512)
# # #             nn.ConvTranspose2d(
# # #                 latent_vector, 1024, kernel_size=4, stride=1, padding=0, bias=False
# # #             ),
# # #             nn.BatchNorm2d(1024),
# # #             nn.ReLU(True),
# # #             # First layer with more channels and kernel size 4
# # #             nn.ConvTranspose2d(
# # #                 1024, 512, kernel_size=4, stride=2, padding=1, bias=False
# # #             ),
# # #             nn.BatchNorm2d(512),
# # #             nn.ReLU(True),
# # #             # Second layer with increased feature map size
# # #             nn.ConvTranspose2d(
# # #                 512, 256, kernel_size=4, stride=2, padding=1, bias=False
# # #             ),
# # #             nn.BatchNorm2d(256),
# # #             nn.ReLU(True),
# # #             # Third layer, larger feature maps and kernels
# # #             nn.ConvTranspose2d(
# # #                 256, 128, kernel_size=4, stride=2, padding=1, bias=False
# # #             ),
# # #             nn.BatchNorm2d(128),
# # #             nn.ReLU(True),
# # #             # Fourth layer with higher resolution output
# # #             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.BatchNorm2d(64),
# # #             nn.ReLU(True),
# # #             # Final layer that outputs an image (output size 64x64 for instance)
# # #             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.Tanh(),  # Tanh to scale the output to [-1, 1]
# # #         )

# # #     def forward(self, z):
# # #         # Reshape the latent vector z into a 4D tensor [batch_size, 512, 1, 1]
# # #         z = z.view(
# # #             z.size(0), 512, 1, 1
# # #         )  # Reshaping to make it compatible with ConvTranspose2d layers
# # #         return self.model(z)


# # # class Discriminator(nn.Module):
# # #     """Current model setup is too complex for a small dataset to train on,"""
# # #     def __init__(self):
# # #         super(Discriminator, self).__init__()

# # #         # More powerful discriminator with more layers and larger channels
# # #         self.model = nn.Sequential(
# # #             # Input is a 64x64 image (or whatever size you are generating)
# # #             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.LeakyReLU(0.2, inplace=True),
# # #             # Second layer with more filters
# # #             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.BatchNorm2d(128),
# # #             nn.LeakyReLU(0.2, inplace=True),
# # #             # Third layer
# # #             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.BatchNorm2d(256),
# # #             nn.LeakyReLU(0.2, inplace=True),
# # #             # Fourth layer
# # #             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
# # #             nn.BatchNorm2d(512),
# # #             nn.LeakyReLU(0.2, inplace=True),
# # #             # Final layer to output a single probability value (real or fake)
# # #             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
# # #         )

# # #     def forward(self, x):
# # #         return self.model(x)
