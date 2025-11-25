import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root)
import json
import torch
import random
import numpy as np
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModel

from diffusers import QwenImageEditPipeline
from torchvision.transforms import CenterCrop

PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

# ========== Utility ==========
def set_seed(seed, rank, device_specific=True):
    """Deterministically set random seed on all relevant libraries."""
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_gpu_env(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
    )
    return args


# ========== Model Init ==========

def initialize_models(args, device, model_path):
    pipeline = QwenImageEditPipeline.from_pretrained(model_path)
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)

    return pipeline


# ========== Core inference ==========

def run_model_and_return_samples(args, model, prompt_text, image=None):
    image = Image.open(image).convert("RGB")

    image = resize_img(image)

    inputs = {
        "image": image,
        "prompt": prompt_text,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "height": image.height,
        "width": image.width,
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = model(**inputs)
        image = output.images[0]

    return image

def resize_img(image):
    image_width, image_height = image.width, image.height
    aspect_ratio = image_width / image_height

    _, target_w, target_h = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_QWENIMAGE_RESOLUTIONS
    )

    scale = max(target_w / image_width, target_h / image_height)
    new_w = int(round(image_width * scale))
    new_h = int(round(image_height * scale))

    image = image.resize((new_w, new_h), Image.LANCZOS)

    crop = CenterCrop((target_h, target_w))
    image = crop(image)

    assert image.width == target_w and image.height == target_h

    return image


# ========== Main ==========

def main(args):
    args = init_gpu_env(args)

    set_seed(args.seed, rank=args.local_rank, device_specific=True)
    device = torch.cuda.current_device()
    model = initialize_models(args, device, model_path=args.model_path)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts for IMGEDIT task
    with open(args.unicbench_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            item = json.loads(line.strip())
            data[item["key"]] = item

    if isinstance(args.languages, str):
        args.languages = [args.languages]

    inference_list = []
    for key, value in data.items():
        outpath = args.output_dir
        outpath = os.path.join(outpath, args.model_name)
        os.makedirs(outpath, exist_ok=True)

        img_path = value["image_path"]
        img_path = os.path.join(args.unicbench_dir, img_path)
        subtask = value["subtask"]
        for lang in args.languages:
            prompt = value[lang]
            inference_list.append([prompt, outpath, key, img_path, subtask, lang])

    # shard across GPUs
    inference_list = inference_list[args.local_rank :: args.world_size]

    for prompt, out_dir, key, img_path, subtask, lang in tqdm(inference_list):
        subtask = subtask.replace(' ', '_')
        out_dir = os.path.join(out_dir, subtask, lang)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{key}.png")
        if os.path.exists(out_file):
            continue

        gen_img = run_model_and_return_samples(args, model, prompt, image=img_path)
        gen_img.save(out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive IMGEDIT sampling script")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--unicbench_path", type=str, default=None)
    parser.add_argument("--unicbench_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="flux_kontext")
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "cn"])

    args = parser.parse_args()

    main(args) 