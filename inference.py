# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/inference.py

import argparse
import warnings
import torch
import random
import subprocess
import json
import numpy as np

from train import export_to_video, load_primary_models, handle_memory_attention
from diffusers import DPMSolverMultistepScheduler
from einops import rearrange
from torch.nn.functional import interpolate
from typing import List, Optional
from pipeline.pipeline_stable_diffusion_xl_video import StableDiffusionXLVideoPipeline
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
from uuid import uuid4

def get_video_height(input_file):
    command = ['ffprobe', 
               '-v', 'quiet', 
               '-print_format', 'json', 
               '-show_streams', 
               input_file]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    for stream in video_info.get('streams', []):
        if stream['codec_type'] == 'video':
            return stream['height']

    return None

def encode_video(input_file, output_file, height):
    command = ['ffmpeg',
               '-i', input_file,
               '-c:v', 'libx264',
               '-crf', '23',
               '-preset', 'fast',
               '-c:a', 'aac',
               '-b:a', '128k',
               '-movflags', '+faststart',
               '-vf', f'scale=-1:{height}',
               '-y',
               output_file]
    
    subprocess.run(command, check=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_pipeline(
    model: str,
    device: str = "cuda"
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        noise_scheduler, tokenizer, text_encoder, tokenizer_2, text_encoder_2, vae, unet = load_primary_models(model, False)

    pipe = StableDiffusionXLVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=noise_scheduler,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        text_encoder_2=text_encoder_2.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()

    return pipe

def prepare_input_latents(
    pipe: StableDiffusionXLVideoPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents

def encode(pipe: StableDiffusionXLVideoPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents

def decode(pipe: StableDiffusionXLVideoPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()

def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

@torch.inference_mode()
def diffuse(
    pipe: StableDiffusionXLVideoPipeline,
    latents: Tensor,
    init_weight: float,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
    height: int,
    width: int,
    rotate: bool
):
    device = pipe.device
    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size, _, num_frames, _, _ = latents.shape
    window_size = min(num_frames, window_size)

    original_size = (height, width)
    target_size = (height, width)

    num_images_per_prompt = 1

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )

    # set the scheduler to start at the correct timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_step = round(init_weight * len(pipe.scheduler.timesteps))
    timesteps = pipe.scheduler.timesteps[start_step:]
    if init_weight == 0:
        latents = torch.randn_like(latents)
    else:
        latents = pipe.scheduler.add_noise(
            original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
        )

    # prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = pipe._get_add_time_ids(
        original_size, (0, 0), target_size, dtype=prompt_embeds.dtype
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # manually track previous outputs for the scheduler as we continually change the section of video being diffused
    model_outputs = [None] * order

    if rotate:
        shifts = np.random.permutation(primes_up_to(window_size))
        total_shift = 0

    with pipe.progress_bar(total=len(timesteps) * num_frames // window_size) as progress:
        for i, t in enumerate(timesteps):
            progress.set_description(f"Diffusing timestep {t}...")

            if rotate:
                shift = shifts[i % len(shifts)]
                model_outputs = [None if pl is None else torch.roll(pl, shifts=shift, dims=2) for pl in model_outputs]
                latents = torch.roll(latents, shifts=shift, dims=2)
                total_shift += shift

            new_latents = torch.zeros_like(latents)
            new_outputs = torch.zeros_like(latents)

            for idx in range(0, num_frames, window_size):
                pipe.scheduler.model_outputs = [model_outputs[(i - 1 - o) % order] for o in reversed(range(order))]
                pipe.scheduler.model_outputs = [
                    None if mo is None else mo[:, :, idx : idx + window_size, :, :].to(device)
                    for mo in pipe.scheduler.model_outputs
                ]
                pipe.scheduler.lower_order_nums = min(i, order)

                latents_window = latents[:, :, idx : idx + window_size, :, :].to(pipe.device)

                latent_model_input = torch.cat([latents_window] * 2) if do_classifier_free_guidance else latents_window
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, text_embeds=add_text_embeds, time_ids=add_time_ids).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                pipe.scheduler.model_outputs = [
                    None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                    for mo in pipe.scheduler.model_outputs
                ]
                latents_window = rearrange(latents_window, "b c f h w -> (b f) c h w")
                noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

                latents_window = pipe.scheduler.step(noise_pred, t, latents_window).prev_sample

                latents_window = rearrange(latents_window, "(b f) c h w -> b c f h w", b=batch_size)

                new_latents[:, :, idx : idx + window_size, :, :] = latents_window.cpu()

                new_outputs[:, :, idx : idx + window_size, :, :] = rearrange(
                    pipe.scheduler.model_outputs[-1], "(b f) c h w -> b c f h w", b=batch_size
                )

                progress.update()

            latents = new_latents
            model_outputs[i % order] = new_outputs

    if rotate:
        new_latents = torch.roll(new_latents, shifts=-total_shift, dims=2)

    return new_latents


@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    window_size: Optional[int] = None,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    init_video: Optional[str] = None,
    init_weight: float = 0.5,
    device: str = "cuda",
    loop: bool = False,
    seed: Optional[int] = None,
    xformers: bool = False,
    sdp: bool = True
):
    if seed is not None:
        set_seed(seed)

    with torch.autocast(device, dtype=torch.half):
        pipe = initialize_pipeline(model, device)

        handle_memory_attention(xformers, sdp, pipe.unet)

        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            init_video=init_video,
            vae_batch_size=vae_batch_size,
        )
        init_weight = init_weight if init_video is not None else 0  # ignore init_weight as there is no init_video!

        latents = diffuse(
            pipe=pipe,
            latents=init_latents,
            init_weight=init_weight,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            height=height,
            width=width,
            rotate=loop or window_size < num_frames
        )

        videos = decode(pipe, latents, vae_batch_size)

    return videos


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=256, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=256, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-WS", "--window-size", type=int, default=None, help="Number of frames to process at once (defaults to full sequence). When less than num_frames, a round robin diffusion process is used to denoise the full sequence iteratively one window at a time. Must be divide num_frames exactly!")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=25, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-i", "--init-video", type=str, default=None, help="Path to video to initialize diffusion from (will be resized to the specified num_frames, height, and width).")
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5, help="Strength of visual effect of init_video on the output (lower values adhere more closely to the text prompt, but have a less recognizable init_video).")
    parser.add_argument("-f", "--fps", type=int, default=12, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-l", "--loop", action="store_true", help="Make the video loop (by rotating frame order during diffusion).")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed to make generations reproducible.")
    args = parser.parse_args()

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    args.prompt = [args.prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size

    if args.window_size is None:
        args.window_size = args.num_frames

    if args.init_video is not None:
        vr = decord.VideoReader(args.init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args.num_frames, args.height, args.width), mode="trilinear")
        args.init_video = init

    # =========================================
    # ============= sample videos =============
    # =========================================

    videos = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        window_size=args.window_size,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        init_video=args.init_video,
        init_weight=args.init_weight,
        device=args.device,
        loop=args.loop,
        seed=args.seed,
        xformers=args.xformers,
        sdp=args.sdp
    )

    # =========================================
    # ========= write outputs to file =========
    # =========================================
    
    for video in videos:
        video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
        video = video.byte().cpu().numpy()
        
        unique_id = str(uuid4())[:8]
        out_file = f"{args.prompt}-{unique_id}.mp4"
        encoded_out_file = f"{args.prompt}-{unique_id}_encoded.mp4"

        export_to_video(video, out_file, args.fps)

        try:
            encode_video(out_file, encoded_out_file, get_video_height(out_file))
        except:
            pass
