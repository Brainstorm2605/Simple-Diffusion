# Simple Stable Diffusion Image Generator with Gradio

This is a simple image generator built using the Stable Diffusion model and the Gradio library for a user-friendly interface. It uses the DDPM sampler for image generation. 

## Features

- **Text-to-Image Generation:** Generate images from text prompts.
- **Image-to-Image Generation:** Modify existing images using text prompts.
- **CFG Scale Control:** Adjust the influence of the text prompt on the generated image.
- **Inference Steps Control:** Control the number of steps in the DDPM sampling process.
- **Random Seed Generation:** A new random seed is used for each image, ensuring diverse results.
- **Negative Prompts:** Refine image generation by specifying what you *don't* want to see in the image.

## Requirements

- Python 3.7+
- PyTorch 
- transformers
- pillow
- gradio
- A Stable Diffusion checkpoint file (e.g., `v1-5-pruned-emaonly.ckpt`)
- CLIP tokenizer files (`tokenizer_vocab.json` and `tokenizer_merges.txt`)

You can install the required Python packages using:

```bash
python -m venv venv
then
venv/Scripts/Activate
and lastly
pip install -r requirements.txt

To Run:
python gradio_app.py 
