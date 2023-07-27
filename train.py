# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/train.py

import argparse
import datetime
import logging
import inspect
import math
import os
import gc
import shutil
import deepspeed
import json
import random
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
import torch.distributed as dist
import subprocess

from typing import Dict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange
from utils.dataset import VideoFolderDataset
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from models.unet import StableDiffusionXLVideoUnet3D
from pipeline.pipeline_stable_diffusion_xl_video import StableDiffusionXLVideoPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils import export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor, LoRAXFormersAttnProcessor, LoRAAttnProcessor2_0

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            for m in module:
                if isinstance(m, BasicTransformerBlock):
                    set_processors([m.attn1, m.attn2])
                    optim_count += 1

    print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def read_deepspeed_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def handle_temporal_params(model, is_enabled=True):
    unfrozen_params = 0

    for name, module in model.named_modules():
        if 'temp_attentions' in name or 'temporal_conv' in name:
            for m in module.parameters():
                m.requires_grad_(is_enabled)
                if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0:
        print(f"{unfrozen_params} params have been unfrozen for training.")

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

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    return out_dir

def load_primary_models(pretrained_model_path, train_2d_model):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")

    if train_2d_model:
        unet = StableDiffusionXLVideoUnet3D.from_pretrained_2d(pretrained_model_path, subfolder="unet")

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    else:
        unet = StableDiffusionXLVideoUnet3D.from_pretrained(pretrained_model_path, subfolder="unet")

        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    print(f'The model has {count_parameters(unet):,} trainable parameters')

    return noise_scheduler, tokenizer, text_encoder, tokenizer_2, text_encoder_2, vae, unet

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")

    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = vae.config.scaling_factor * latents

    return latents

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1) and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        unet, 
        vae, 
        text_encoder, 
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        scheduler,
        output_dir,
        is_checkpoint=False
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    pipeline = StableDiffusionXLVideoPipeline.from_pretrained(path, 
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler)
    
    pipeline.save_pretrained(save_path)

    print(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline

    torch.cuda.empty_cache()
    gc.collect()

def save_unet(global_step, unet, output_dir, is_checkpoint=False):
    save_path = output_dir
    
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        torch.save(unet.state_dict(), os.path.join(save_path, 'diffusion_pytorch_model.bin'))

        existing_checkpoints = [d for d in os.listdir(output_dir) if 'checkpoint-' in d]
        existing_checkpoints = sorted(existing_checkpoints, key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))

        # We keep only the last 3 checkpoints
        while len(existing_checkpoints) > 3:
            shutil.rmtree(os.path.join(output_dir, existing_checkpoints.pop(0)))
    else:
        torch.save(unet.state_dict(), os.path.join(save_path, 'diffusion_pytorch_model.bin'))

    logging.info(f"Saved unet at {save_path} on step {global_step}")

    torch.cuda.empty_cache()
    gc.collect()

def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 

def main(
    pretrained_2d_model_path: str,
    pretrained_3d_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    train_2d_model: bool = True,
    epochs: int = 1,
    validation_steps: int = 100,
    checkpointing_steps: int = 500,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    use_offset_noise: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
    enable_torch_2_attn: bool = True,
    offset_noise_strength: float = 0.1,
    **kwargs
):
    dist.init_process_group(backend='nccl')

    *_, config = inspect.getargvalues(inspect.currentframe())

    if dist.get_rank() == 0:
        output_dir = create_output_folders(output_dir, config)

    if train_2d_model:
        pretrained_model_path = pretrained_2d_model_path
    else:
        pretrained_model_path = pretrained_3d_model_path

    noise_scheduler, tokenizer, text_encoder, tokenizer_2, text_encoder_2, vae, unet = load_primary_models(pretrained_model_path, train_2d_model)

    data = read_deepspeed_config_file(train_data.deepspeed_config_file)

    unet_engine, _, _, _ = deepspeed.initialize(
        model=unet,
        model_parameters=unet.parameters(),
        config=train_data.deepspeed_config_file,
    )

    text_encoder.to(unet_engine.device)
    text_encoder_2.to(unet_engine.device)
    vae.to(unet_engine.device)
    unet.to(unet_engine.device)

    freeze_models([text_encoder, text_encoder_2, vae, unet])

    vae.enable_slicing()

    train_dataset = VideoFolderDataset(**train_data, tokenizer=tokenizer, text_encoder=text_encoder, tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2, unet=unet, vae=vae, device=unet_engine.device)

    train_sampler = DistributedSampler(train_dataset, shuffle=False, seed=seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data['train_micro_batch_size_per_gpu'],
        sampler=train_sampler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    global_step = 0

    unet.train()

    if gradient_checkpointing:
        unet._set_gradient_checkpointing(value=True)

    handle_temporal_params(unet, is_enabled=True)

    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)
    
    progress_bar = tqdm(range(global_step, num_update_steps_per_epoch * epochs))
    progress_bar.set_description("Steps")

    def finetune_unet(batch):
        pixel_values = batch["pixel_values"].to(unet_engine.device)
        
        latents = tensor_to_vae_latent(pixel_values, vae)

        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = batch['prompt_embeds'].to(unet_engine.device)
        text_embeds = batch['add_text_embeds'].to(unet_engine.device)
        time_ids = batch['add_time_ids'].to(unet_engine.device)

        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, text_embeds=text_embeds, time_ids=time_ids).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    for _ in range(0, epochs):
        for batch in train_dataloader:
            with autocast():
                loss = finetune_unet(batch)

            unet_engine.backward(loss)
            unet_engine.step()

            if dist.get_rank() == 0:
                progress_bar.update(1)
                global_step += 1

                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        unet,
                        vae,
                        text_encoder,
                        tokenizer,
                        text_encoder_2,
                        tokenizer_2,
                        noise_scheduler,
                        output_dir,
                        is_checkpoint=True
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    with autocast():
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=False)
                        unet.eval()

                        pipeline = StableDiffusionXLVideoPipeline.from_pretrained(pretrained_model_path, 
                                                                            unet=unet,
                                                                            vae=vae,
                                                                            text_encoder=text_encoder,
                                                                            text_encoder_2=text_encoder_2,
                                                                            tokenizer=tokenizer,
                                                                            tokenizer_2=tokenizer_2)
                        
                        diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                        pipeline.scheduler = diffusion_scheduler

                        prompt = random.choice(batch["text_prompt"]) if len(validation_data.prompt) <= 0 else validation_data.prompt

                        save_filename = f"{global_step}-{prompt}"
                        out_file = f"{output_dir}/samples/{save_filename}.mp4"
                        encoded_out_file = f"{output_dir}/samples/{save_filename}_encoded.mp4"

                        with torch.no_grad():
                            video_frames = pipeline(
                                prompt,
                                device=unet_engine.device,
                                width=validation_data.width,
                                height=validation_data.height,
                                num_frames=validation_data.num_frames,
                                num_inference_steps=validation_data.num_inference_steps,
                                guidance_scale=validation_data.guidance_scale
                            ).frames

                        export_to_video(video_frames, out_file, validation_data.fps)

                        try:
                            encode_video(out_file, encoded_out_file, get_video_height(out_file))
                        except:
                            pass
                            
                        del pipeline, video_frames
                        torch.cuda.empty_cache()
                        
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=True)
                        unet.train()

    if dist.get_rank() == 0:
        save_pipe(
            pretrained_model_path,
            global_step,
            unet,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            noise_scheduler,
            output_dir,
            is_checkpoint=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training.yaml")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank of this process. Used for distributed training.')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
