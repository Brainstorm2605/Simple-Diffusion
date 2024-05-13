# Simple Stable Diffusion Image Generator with Gradio

This is a simple image generator built using the Stable Diffusion model and the Gradio library for a user-friendly interface. It uses the DDPM sampler for image generation. 

![Image Description](https://github.com/Brainstorm2605/Diffusion/blob/master/Image/Screenshot%202024-05-13%20145259.png)

## Features

- **Text-to-Image Generation:** Generate images from text prompts.
- **Image-to-Image Generation:** Modify existing images using text prompts.
- **CFG Scale Control:** Adjust the influence of the text prompt on the generated image.
- **Inference Steps Control:** Control the number of steps in the DDPM sampling process.
- **Random Seed Generation:** A new random seed is used for each image, ensuring diverse results.
- **Negative Prompts:** Refine image generation by specifying what you *don't* want to see in the image.

## Requirements

- Python 3.7+
- PyTorch with cuda
- transformers
- pillow
- gradio
- A Stable Diffusion checkpoint file v1-5-pruned-emaonly.ckpt ( https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt )

You can install the required Python packages using:

```bash
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt

To Run:
python gradio_app.py 
