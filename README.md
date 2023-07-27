# Stable Diffusion XL Video
## Train a Video Diffusion Model Using Stable Diffusion XL Image Priors

This project involves training a video diffusion model based on Stable Diffusion XL image priors. It is heavily inspired by the "Make-a-video" paper by Meta. For more details, please refer to this link: (https://arxiv.org/abs/2209.14792)

## Getting Started

### Installation
```bash
git clone https://github.com/motexture/stable-diffusion-xl-video.git
cd stable-diffusion-xl-video
```

### Python Requirements

```bash
pip install deepspeed
pip install -r requirements.txt
```

On some systems, deepspeed requires installing the CUDA toolkit first in order to properly install. If you do not have CUDA toolkit, or deepspeed shows an error follow the instructions by NVIDIA: https://developer.nvidia.com/cuda-downloads

or on linux systems:
```bash
sudo apt install build-essential
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

During the installation you only need to install toolkit, not the drivers or documentation.


## Preparing the config file
Open the training.yaml file and modify the parameters according to your needs.

## Train
```python
deepspeed train.py --config training.yaml
```
---

## Running inference
The `inference.py` script can be used to render videos with trained checkpoints.

Example usage: 
```
python inference.py \
  --model sdxlvid \
  --prompt "a fast moving fancy sports car" \
  --num-frames 16 \
  --width 1024 \
  --height 1024 \
  --sdp
```

## Shoutouts

- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for the original training and inference code
- [lucidrains](https://github.com/lucidrains/make-a-video-pytorch/) for the pseudo 3D convolutions and the "make-a-video" implementation
- [xuduo35](https://github.com/xuduo35/MakeLongVideo/) for his own "make-a-video" implementation
- [guoyww](https://github.com/guoyww/AnimateDiff/) for the AnimateDiff paper and code
- [Showlab](https://github.com/showlab/Tune-A-Video) and [bryandlee](https://github.com/bryandlee/Tune-A-Video) for their Tune-A-Video contribution
