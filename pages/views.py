import base64
import glob
import io

import numpy as np
import torch

# Create your views here.
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.generic import TemplateView
from PIL import Image

from gan.models import Generator

# gan_app/views.py


# Initialize the Generator (loaded from saved model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(model, model_type="generator"):
    # Find all the checkpoint files for the model type (generator or discriminator)
    checkpoint_files = glob.glob(f"gan/models/*{model_type}.pth")
    if not checkpoint_files:
        print(f"No {model_type} checkpoint found!")
        return

    # Sort files by epoch number (latest first) using a similar pattern
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)
    latest_checkpoint = checkpoint_files[0]

    # Load the latest checkpoint file
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {model_type} model from {latest_checkpoint}")


generator = Generator().to(device)
load_model_from_checkpoint(generator, model_type="generator")

# load_model_from_checkpoint(discriminator, model_type="discriminator")


def image_gen(request):
    try:
        noise = torch.randn(1, 100).to(device)  # Latent vector
        generated_image = generator(noise).cpu().detach()

        # Convert tensor to image and save as PNG
        image = generated_image.squeeze().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Save to a buffer
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        return JsonResponse({"image": encoded_image})

    except Exception as e:
        return JsonResponse({"error": f"Error generating image: {str(e)}"}, status=500)


@method_decorator(never_cache, name="dispatch")
class HomePageView(TemplateView):
    template_name = "home.html"
