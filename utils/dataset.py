# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/utils/dataset.py

import os
import decord
import numpy as np
import random
import torchvision.transforms as T
import torch
import cv2
import concurrent.futures

from tqdm import tqdm
from typing import Optional
from .bucketing import sensible_buckets
from diffusers.loaders import TextualInversionLoaderMixin

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

def process_file(args):
    file, root = args
    if file.endswith('.mp4'):
        return os.path.join(root, file)
    
    return None
    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)
    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr
 
class VideoFolderDataset(Dataset):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        frame_step: int = 4,
        path: str = "./data",
        use_bucketing: bool = False,
        tokenizer = None,
        tokenizer_2 = None,
        text_encoder = None,
        text_encoder_2 = None,
        unet = None,
        vae = None,
        device = "cuda",
        **kwargs
    ):
        self.use_bucketing = use_bucketing

        self.video_files = []
        self.find_videos(path)
        
        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae = vae

        self.device = device
    
    def find_videos(self, path):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    jobs.append(executor.submit(process_file, (file, root)))
            
            for future in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs)):
                result = future.result()
                if result is not None:
                    self.video_files.append(result)

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()

        every_nth_frame = max(1, round(self.frame_step * native_fps / 30))

        if len(vr) < n_sample_frames * every_nth_frame:
            return None, None

        effective_length = len(vr) // every_nth_frame

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)

        if video.shape[-1] == 4:
            video = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in video])

        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: 
            video = resize(video)
                
        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
            vid_path,
            self.use_bucketing,
            self.width, 
            self.height, 
            self.get_frame_buckets, 
            self.get_frame_batch
        )

        return video, vr
        
    def get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    ):
        device = device or self.device

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape

                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        try:
            video, _ = self.process_video_wrapper(self.video_files[index])
        except:
            return self.__getitem__((index + 1) % len(self))
        
        if video is None or (video and video[0] is None):
            return self.__getitem__((index + 1) % len(self))

        basename = os.path.basename(self.video_files[index]).replace('.mp4', '').replace('_', ' ')
        split_basename = basename.split('-')

        if len(split_basename) > 1:
            prompt = '-'.join(split_basename[:-1])
        else:
            prompt = split_basename[0]

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=prompt, device=self.device)

        original_size = (self.height, self.width)
        target_size = (self.height, self.width)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.get_add_time_ids(
            original_size, (0, 0), target_size, dtype=prompt_embeds.dtype
        )

        return {"pixel_values": (video[0] / 127.5 - 1.0), "prompt_embeds": prompt_embeds[0], "add_text_embeds": add_text_embeds[0], "add_time_ids": add_time_ids[0], "text_prompt": prompt, "file": basename, 'dataset': self.__getname__()}