import base64
import glob
import io
import os
import time

import numpy as np
import torch

# Create your views here.
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.generic import TemplateView
from PIL import Image

from gan.models import Discriminator, Generator
from gan.utils import latent_vector

# Initialize the Generator (loaded from saved model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load a model from the latest checkpoint
def load_model_from_checkpoint(model, model_type="generator", device=None):
    try:
        # Find all the checkpoint files for the model type (generator or discriminator)
        checkpoint_files = glob.glob(f"gan/models/*{model_type}.pth")
        if not checkpoint_files:
            print(f"No {model_type} checkpoint found!")
            return

        # Sort files by epoch number (latest first)
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        # Load the checkpoint file
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {model_type} model from {latest_checkpoint}")

    except Exception as e:
        print(f"Error loading {model_type} model: {str(e)}")


# Load the models globally
generator = Generator().to(device)
load_model_from_checkpoint(generator, model_type="generator", device=device)

discriminator = Discriminator().to(device)
load_model_from_checkpoint(discriminator, model_type="discriminator", device=device)


# Simple mapping of text to labels (extend for more complex text)
def text_to_label(text):
    text = text.lower().strip()
    if "cat" in text:
        return torch.tensor([0]).long()  # Cat label (0)
    elif "dog" in text:
        return torch.tensor([1]).long()  # Dog label (1)
    return torch.tensor([0]).long()  # Default to cat if unclear


def image_gen(request):
    try:
        # Get text input from the POST request
        text = request.POST.get("text", "random cat")  # Default to "random cat"
        # Convert text to label (0 for cat, 1 for dog, or extend for text embeddings)
        labels = text_to_label(text).to(device)
        # Generate random latent vector for the generator
        noise = torch.randn(1, latent_vector).to(device)

        with torch.no_grad():
            generated_image = generator(noise, labels).cpu().detach()

        # Process the generated image (convert to base64, etc.)
        image = generated_image.squeeze().permute(1, 2, 0).numpy()
        image = ((image + 1) * 127.5).astype(
            np.uint8
        )  # Rescale from [-1, 1] to [0, 255]
        pil_image = Image.fromarray(image)

        os.makedirs("datasets/generated_images", exist_ok=True)

        # Save the image locally with a timestamp
        timestamp = int(time.time())
        image_path = f"datasets/generated_images/generated_image_{timestamp}.png"
        pil_image.save(image_path)

        # Save the image to a buffer and encode it in base64
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        # Return the base64 encoded image as JSON
        return JsonResponse({"image": encoded_image})

    except Exception as e:
        return JsonResponse({"error": f"Error generating image: {str(e)}"}, status=500)


@method_decorator(never_cache, name="dispatch")
class HomePageView(TemplateView):
    template_name = "home.html"
