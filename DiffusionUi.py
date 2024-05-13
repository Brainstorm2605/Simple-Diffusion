import gradio as gr
from PIL import Image
import torch
import model_loader
import pipeline
from transformers import CLIPTokenizer
import random

# Set up device
DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = True
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"


# Load tokenizer and models
tokenizer = CLIPTokenizer(
    "Data/tokenizer_vocab.json", merges_file="Data/tokenizer_merges.txt"
)
model_file = "Data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Gradio Interface
def generate_image(prompt, negative_prompt, cfg_scale, inference_steps, uploaded_image, strength):
    # Generate seed on the go
    seed = random.randint(0, 1000000)
    
    # Prepare input image
    input_image = None
    if uploaded_image is not None:
        input_image = Image.open(uploaded_image)

    # Generate image
    image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    return Image.fromarray(image), seed # Return the image and the seed

inputs = [
    gr.Textbox(label="Prompt", value="A cat wearing sunglasses"),
    gr.Textbox(
        label="Negative Prompt (optional)",
        value="BadDream, UnrealisticDream, lowres, bad anatomy, bad hands, text, error, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)),paintings",
    ),
    gr.Slider(label="CFG Scale", minimum=1.0, maximum=14.0, value=5.0, step=0.5),
    gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=50, step=1),
    gr.Image(label="Upload an Image (optional)", type="pil"),
    gr.Slider(
        label="Image-to-Image Strength", minimum=0.0, maximum=1.0, value=0.9, step=0.1
    ),
]

outputs = [gr.Image(label="Generated Image"), gr.Number(label="Seed")] # Display seed

iface = gr.Interface(
    fn=generate_image,
    inputs=inputs,
    outputs=outputs,
    title="SimpleImage Generator",
    # Rename the submit button to "Generate"
    live=False,
    allow_flagging="never",
)

iface.launch()