import torch
import json
import math
import os
import torch.multiprocessing as mp
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

MODEL_PATH_CONTROLNET = "FLUX.1-dev-Controlnet-Inpainting-Beta"
MODEL_PATH_BASE = "FLUX.1-dev"
JSON_PATH = "complex-images-synthetic/data_with_selected_mask.json"
NUM_GPUS = 4  

def run_worker(rank):

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    chunk_size = len(data) // NUM_GPUS
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != NUM_GPUS - 1 else len(data)
    data_split = data[start_idx:end_idx]

    controlnet = FluxControlNetModel.from_pretrained(MODEL_PATH_CONTROLNET, torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_PATH_BASE, subfolder='transformer', torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        MODEL_PATH_BASE,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)

    for item in data_split:
        image_path = item['image']
        mask_path = item['selected_object']['mask']
        save_path = f"complex-images-synthetic/inpainting_images/{item['id']}.png"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        prompt = 'background'
        negative_prompt = item['selected_object']['class_name']

        width, height = 1024, 1024
        size = (math.ceil(width / 8) * 8, math.ceil(height / 8) * 8)
        try:
            image = load_image(image_path).convert("RGB").resize(size)
            mask = load_image(mask_path).convert("RGB").resize(size)
        except Exception as e:
            continue
        generator = torch.Generator(device=device).manual_seed(24)

        result = pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt=negative_prompt,
            true_guidance_scale=1.0
        ).images[0]

        result.save(save_path)


if __name__ == "__main__":
    mp.spawn(run_worker, nprocs=NUM_GPUS)

