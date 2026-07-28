import os
import sys
import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_2d_condition import UNet2DConditionModel
from musepose.models.unet_3d import UNet3DConditionModel
from musepose.pipelines.pipeline_pose2img import Pose2ImagePipeline

# Shared profiling utility (self-contained, copied to this directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiling_utils import BenchmarkProfiler

# Resolve project root so the script can be run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "test_stage_1.yaml"))
    # Benchmark overrides: W=256, H=176, steps=50 (original defaults were 768/768/20)
    parser.add_argument("-W", type=int, default=256)
    parser.add_argument("-H", type=int, default=176)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnt", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=7)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Run from project root so relative paths in the config resolve correctly.
    os.chdir(PROJECT_ROOT)

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to("cuda", dtype=weight_dtype)

    # ------------------------------------------------------------------
    # Profiling: record parameter count and size for every model.
    # ------------------------------------------------------------------
    profiler = BenchmarkProfiler(
        project_name="Musepose_copy1",
        log_dir=BENCHMARK_RESULTS_DIR,
    )
    profiler.results["image_size"] = f"{args.W}x{args.H}"
    profiler.results["denoising_steps"] = args.steps
    profiler.results["cfg"] = args.cfg
    profiler.results["seed"] = args.seed
    profiler.results["weight_dtype"] = config.weight_dtype

    profiler.record_all_models({
        "vae": vae,
        "reference_unet": reference_unet,
        "denoising_unet": denoising_unet,
        "pose_guider": pose_guider,
        "image_enc": image_enc,
    })

    # ------------------------------------------------------------------
    # Pick the first test case for a single benchmark inference.
    # ------------------------------------------------------------------
    first_ref_dir = list(config["test_cases"].keys())[0]
    if os.path.isdir(first_ref_dir):
        ref_image_paths = glob.glob(os.path.join(first_ref_dir, '*.jpg'))
    else:
        ref_image_paths = [first_ref_dir]
    ref_image_path = ref_image_paths[0]

    first_pose_dir = config["test_cases"][first_ref_dir][0]
    if os.path.isdir(first_pose_dir):
        pose_image_paths = glob.glob(os.path.join(first_pose_dir, '*.jpg'))
    else:
        pose_image_paths = [first_pose_dir]
    pose_image_path = pose_image_paths[0]

    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    pose_image = Image.open(pose_image_path).convert("RGB")

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    pose_image_tensor = pose_transform(pose_image).unsqueeze(0)
    ref_image_tensor = pose_transform(ref_image_pil).unsqueeze(1).unsqueeze(0)

    print("=" * 60)
    print(f"[BENCHMARK] project     : Musepose_copy1")
    print(f"[BENCHMARK] steps       : {args.steps}")
    print(f"[BENCHMARK] image size  : {args.W}x{args.H}")
    print(f"[BENCHMARK] ref image   : {ref_image_path}")
    print(f"[BENCHMARK] pose image  : {pose_image_path}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Inference with profiling.
    # BenchmarkProfiler.__enter__ syncs CUDA, starts the timer and resets
    # GPU peak-memory stats; __exit__ records inference_time_seconds and
    # peak_gpu_memory_mb.
    # ------------------------------------------------------------------
    generator = torch.manual_seed(args.seed)

    with profiler:
        image = pipe(
            ref_image_pil,
            pose_image,
            width,
            height,
            args.steps,
            args.cfg,
            generator=generator,
        ).images

    # Save a sanity-check output image next to the benchmark log.
    image = image.squeeze(2).squeeze(0)  # (c, h, w)
    image = image.transpose(0, 1).transpose(1, 2)  # (h, w, c)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    sample_out = os.path.join(
        BENCHMARK_RESULTS_DIR,
        f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    image.save(sample_out)
    profiler.results["sample_image"] = sample_out

    log_file = profiler.save_log()
    print(f"[BENCHMARK] Done. Log saved to: {log_file}")


if __name__ == "__main__":
    main()
