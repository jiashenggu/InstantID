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

from onediffx import compile_pipe, save_pipe, load_pipe

import socket


def get_private_ip():
    # The socket connects to a Google's DNS server IP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s"
)

private_ip = get_private_ip()
if "LOG_NAME" in os.environ:
    log_name = os.environ["LOG_NAME"]
else:
    log_name = "dev"
log_file_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"logs/instantid_{log_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{private_ip}.log",
)
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

file_handler = logging.FileHandler(log_file_name, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    
    def flush(self):
        pass

    def isatty(self):
        return False

# Redirect standard output and standard error to the log file
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


device = "cuda"

face_analysis = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_analysis.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
face_image = Image.open('/ML-A100/team/mm/gujiasheng/InstantID/examples/obama.jpg')
face_analysis.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

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

if "OUTPUT_SIZE" in os.environ:
    size = os.environ["OUTPUT_SIZE"]
else:
    size = "1280x1280"

if os.path.exists(f"cached_pipe_{base_model_name}_{size}"):
    load_pipe(pipe, dir=f"cached_pipe_{base_model_name}_{size}")
    logger.info("Load cached pipe: " + f"cached_pipe_{base_model_name}_{size}")

pipe.set_progress_bar_config(disable=True)


app = FastAPI()

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import datetime

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request.state.request_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = await call_next(request)
        end_time = time.time()
        response.headers['X-Request-Start-Time'] = str(start_time)
        response.headers['X-Request-End-Time'] = str(end_time)
        return response
    
app.add_middleware(TimingMiddleware)

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

def pad_to_768_1344(
    input_image,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):

    w, h = input_image.size

    ratio = 768/w
    w_resize_new = 768
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number

    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    res = np.ones([1344, 768, 3], dtype=np.uint8) * 255
    offset_x = (768 - w_resize_new) // 2
    offset_y = (1344 - h_resize_new) // 2
    res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
        np.array(input_image)
    )
    input_image = Image.fromarray(res)

    return input_image

def pad_to_assigned_size(
    input_image,
    w_side=768,
    h_side=1344,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):

    w, h = input_image.size

    ratio = w_side/w
    w_resize_new = w_side
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number

    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    res = np.ones([h_side, w_side, 3], dtype=np.uint8) * 255
    offset_y = (h_side - h_resize_new) // 2
    if offset_y < 0:
        res = np.array(input_image)[:h_side, :]
    else:
        res[offset_y : offset_y + h_resize_new, : w_resize_new] = (
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


class instantid_data(BaseModel):
    base_model_name: str = "LahHongchenSDXLSD15_sdxlV10"
    image_url: str
    image_format: str = "jpg"
    prompt: str = ""
    n_prompt: str = ""
    response_format: str = "b64_json"
    size: str = "1280x1280"
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
    width: int = 1280,
    height: int = 1280,
    prompt: str = "",
    n_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: int = 5,
    seed: int = 42,
):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Inference Start Time: {cur_time}")
    # Infer setting
    if prompt == "":
        prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus,  vibrant, colorful, ultra sharpness, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
    if n_prompt == "":
        n_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, monochrome"


    
    # face_image = resize_img(face_image)
    # expanded_face_image = resize_img(expanded_face_image)

    face_image = face_image.convert("RGB")
    face_image = pad_to_assigned_size(face_image, w_side=width, h_side=height)

    
    face_infos = face_analysis.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if len(face_infos) == 0:
        expanded_face_image = expand_image_with_white(face_image, expand_size=100)
        expanded_face_image = expanded_face_image.convert("RGB")
        expanded_face_image = pad_to_assigned_size(expanded_face_image, w_side=width, h_side=height)
        logger.info("No face detected. Try to use the expanded image with white border.")
        face_infos = face_analysis.get(cv2.cvtColor(np.array(expanded_face_image), cv2.COLOR_RGB2BGR))
        if len(face_infos) == 0:
            raise Exception("After expanding the image with white border. Still no face detected.")
    for face_info in face_infos:
        logger.info("det_score: " + (str(face_info["det_score"])))
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
        width=width,
        height=height,
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
    logger.info(f"generation_time: {time.time() - start}")
    return image


@app.post("/img2img")
async def img2img(request: Request, data: instantid_data):
    api_process_start = time.time()
    base_model_name = data.base_model_name
    image_url = data.image_url
    prompt = data.prompt
    n_prompt = data.n_prompt
    style = data.style
    num_inference_steps = data.num_inference_steps
    guidance_scale = data.guidance_scale
    seed = data.seed
    can_empty = data.can_empty
    width, height = map(int, data.size.split("x"))
    info = data.model_dump()
    hash_id = hash(image_url+prompt+n_prompt+style+str(num_inference_steps)+str(guidance_scale)+str(seed))
    info["hash_id"] = hash_id
    logger.info(f"Request Time: {request.state.request_start_time}")
    if not info['image_url'].startswith('http'):
        del info['image_url']
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
    
    
    filename = f"{hash_id}"

    output_image = await process_image(
        input_image, width, height, prompt, n_prompt, num_inference_steps, guidance_scale, seed
    )

    if not os.path.exists(f"cached_pipe_{base_model_name}_{data.size}"):
        save_pipe(pipe, dir=f"cached_pipe_{base_model_name}_{data.size}")
        logger.info("Save cached pipe: " + f"cached_pipe_{base_model_name}_{data.size}")
    private_ip = get_private_ip()
    info["private_ip"] = private_ip

    api_process_end = time.time()
    logger.info(f"api_process_time: {api_process_end - api_process_start}")
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


