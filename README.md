# Django GAN
This GAN project is subject to change and should not be considered a full project just yet, 

iterations :
  - GAN
  - WGAN
  - WGAN-GP (current setup with Tensorflow)
    
This is a Django web app that allows you to generate images using a pre-trained Generative Adversarial Network (GAN). 

## Features
- Generate images using a trained GAN model.
- Interface for generating random images from a latent vector.
- Built with Django and PyTorch.

## Requirements
- Python >= 3.11
- Django >= 5.1.6
- PyTorch >= 1.12.0
- torchvision >= 0.13.0

## Setup
1. Add .env 
<!-- Im using uv to managed my packages, super easy and simple to use, highly recommend using it -->
run :
$ uv venv   <-- creates the .venv environment that will be used int the project 
$ uv sync   <-- syncs the .venv environment with the pyproject.toml dependency's 


2. Activate you .venv environment. 
<!-- side note - ill working in linux so ill  provide commands for that  -->
$ source .venv/bin/activate

3. Start Django server:
$ python manage.py runserver 8000 
// You don't need to specify a port, default is 8000

there are 

4. To Train
   
First ensure that you have images/ dataset, you want to train the model on, save training images in datasets/images/
in your terminal run :

$ python gan/train.py

this will run the training script on the models 



5. Navigate to the GAN
Go to your browser and start generating image,

localhost:8000

## Notes :

The current setup has moved away from a traditional GAN - Generative Adversarial Network, and is now an implementation of a WGAN - Wasserstein GAN.

this project will most likely go through a few iterations of being setup in different versions of a GAN.
