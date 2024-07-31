import sys

sys.path.append("./")

import numpy as np
import cv2
import torch
import os
import time
import json
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn
from io import BytesIO
import aiohttp
import base64
from PIL import Image, ImageOps
import logging
import datetime
from pydantic import BaseModel, Field
from typing import Tuple
from style_template import styles

from onediffx import compile_pipe, save_pipe, load_pipe

import socket


def get_private_ip():
    # The socket connects to a Google's DNS server IP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a file handler to log to a file
file_handler = logging.FileHandler(
    f'logs/app_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
)
file_handler.setLevel(logging.INFO)

# Add a stream handler to log to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


device = "cuda"

face_analysis = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_analysis.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# base_model_path = '/ML-A100/team/mm/gujiasheng/sd_webui_0/models/Stable-diffusion/animagine-xl-3.0'
base_model_name = "LahHongchenSDXLSD15_sdxlV10"
base_model_path = f"/ML-A100/team/mm/gujiasheng/model/{base_model_name}"
base_model_name = base_model_path.split("/")[-1]
logger.info(f"Model Info: {base_model_path}")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.cuda()
pipe.load_ip_adapter_instantid(face_adapter)
pipe = compile_pipe(pipe)
if os.path.exists(f"cached_pipe_{base_model_name}"):
    load_pipe(pipe, dir=f"cached_pipe_{base_model_name}")


app = FastAPI()
DEFAULT_STYLE_NAME = "(No style)"


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image

def expand_image_with_white(input_image, expand_size=100):
    width, height = input_image.size
    if width < height:
        diff = height - width
        padding = (diff // 2, 0, diff // 2 + diff % 2, 0)
    elif width > height:
        diff = width - height
        padding = (0, diff // 2, 0, diff // 2 + diff % 2)
    else:
        padding = (expand_size, expand_size, expand_size, expand_size)
    expanded_image = ImageOps.expand(input_image, padding, fill=(255, 255, 255))
    logger.info(f"Expanded image size: {expanded_image.size}")
    return expanded_image

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative


class instantid_data(BaseModel):
    base_model_name: str = "LahHongchenSDXLSD15_sdxlV10"
    image_url: str
    image_format: str = "jpg"
    prompt: str = ""
    n_prompt: str = ""
    response_format: str = "b64_json"
    style: str = DEFAULT_STYLE_NAME
    num_inference_steps: int = 30
    guidance_scale: int = 5
    seed: int = 42
    can_empty: bool = False


class ImageToImageResponse(BaseModel):
    images: list[str] = Field(
        default=None, title="Image", description="The generated image in base64 format."
    )
    parameters: dict
    info: str


# This function is a simplified version of your processing script integrated into the API
async def process_image(
    face_image: Image.Image,
    prompt: str = "",
    n_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: int = 5,
    seed: int = 42,
):
    # Infer setting
    if prompt == "":
        prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus,  vibrant, colorful, ultra sharpness, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
    if n_prompt == "":
        n_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, monochrome"

    expanded_face_image = expand_image_with_white(face_image, expand_size=100)
    face_image = resize_img(face_image)
    expanded_face_image = resize_img(expanded_face_image)

    face_infos = face_analysis.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if len(face_infos) == 0:
        logger.info("No face detected. Try to use the expanded image with white border.")
        face_infos = face_analysis.get(cv2.cvtColor(np.array(expanded_face_image), cv2.COLOR_RGB2BGR))
        if len(face_infos) == 0:
            raise Exception("After expanding the image with white border. Still no face detected.")
    for face_info in face_infos:
        logger.info(str(face_info["det_score"]))
    face_info = sorted(
        face_infos,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(face_image, face_info["kps"])

    start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=(
            torch.Generator(device=device).manual_seed(seed) if seed != 42 else None
        ),
    ).images[0]
    logger.info(f"Time: {time.time() - start}")
    return image


@app.post("/img2img")
async def img2img(data: instantid_data):
    base_model_name = data.base_model_name
    image_url = data.image_url
    prompt = data.prompt
    n_prompt = data.n_prompt
    style = data.style
    num_inference_steps = data.num_inference_steps
    guidance_scale = data.guidance_scale
    seed = data.seed
    can_empty = data.can_empty
    info = data.model_dump()
    hash_id = hash(image_url+prompt+n_prompt+style+str(num_inference_steps)+str(guidance_scale)+str(seed))
    info["hash_id"] = hash_id
    logger.info(f"Request Info: \n{json.dumps(info)}")
    if can_empty is True:
        import GPUtil

        GPUs = GPUtil.getGPUs()
        gpu_utility = GPUs[0].load * 100
        logger.info("GPU utility: "+str(gpu_utility))
        if gpu_utility > 5:
            return ImageToImageResponse(
                images=[], parameters=vars(data), info=json.dumps(info)
            )
    if base_model_name != "LahHongchenSDXLSD15_sdxlV10":
        return {"error": "Invalid model name"}
    if image_url.startswith("http://") or image_url.startswith("https://"):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                image_bytes = await response.read()
                input_image = Image.open(BytesIO(image_bytes))
    elif os.path.exists(image_url):
        input_image = Image.open(image_url)
    else:
        image_bytes = base64.b64decode(image_url)
        input_image = Image.open(BytesIO(image_bytes))
    input_image = expand_image_with_white(input_image, expand_size=0)
    
    filename = f"{hash_id}"
    prompt, n_prompt = apply_style(style, prompt, n_prompt)

    output_image = await process_image(
        input_image, prompt, n_prompt, num_inference_steps, guidance_scale, seed
    )

    if not os.path.exists(f"cached_pipe_{base_model_name}"):
        save_pipe(pipe, dir=f"cached_pipe_{base_model_name}")
    private_ip = get_private_ip()
    info["private_ip"] = private_ip
    if data.response_format == "b64_json":
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG")
        b64images = base64.b64encode(buffered.getvalue()).decode()
        return ImageToImageResponse(
            images=[b64images],
            parameters=vars(data),
            info=json.dumps(info),
        )
    elif data.response_format == "file_path":
        if os.path.exists(image_url):
            filename = image_url.split("/")[-1].replace(".jpg", f"_output.jpg")
        else:
            filename = f"{hash_id}.jpg"
        image_path = os.path.abspath(f"output/{filename}")
        output_image.save(image_path)
        return ImageToImageResponse(
            images=[image_path],
            parameters=vars(data),
            info=json.dumps(info),
        )
    else:
        return {"error": "Invalid response format"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
