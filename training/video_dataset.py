import glob
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch


@dataclass
class VAEConfig:
    name: str
    spatial_downsample_factor: int
    temporal_downsample_factor: int
    latent_channels: int


VAE_CONFIGS = {
    "wan": VAEConfig(
        name="wan",
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        latent_channels=16,
    ),
}


@dataclass
class VideoResolution:
    vae_config: VAEConfig
    height: int
    width: int
    num_frames: int

    def __post_init__(self):
        self.num_channels = self.vae_config.latent_channels

    @property
    def latent_frames(self) -> int:
        num_frames = self.num_frames
        lsize = 1 + math.ceil((num_frames - 1) / self.vae_config.temporal_downsample_factor)
        lsize = int(lsize)
        return lsize

    @property
    def latent_height(self) -> int:
        return self.height // self.vae_config.spatial_downsample_factor

    @property
    def latent_width(self) -> int:
        return self.width // self.vae_config.spatial_downsample_factor


def tuplize(x):
    # handle string input
    if isinstance(x, str):
        x = x.split(",")
        return tuple(int(i) for i in x)
    # handle non-iterable input
    if not isinstance(x, Iterable):
        return (x, x)
    return x


class DirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir_path: str,
        vae_name: str,
        resolution: Union[Tuple[int, int], int],
        max_sequence_length: int,
        num_frames: int,
    ):
        self.dir_path = dir_path
        resolution_tuple = tuplize(resolution)  # resolution is (height, width)

        self.video_resolution = VideoResolution(
            height=resolution_tuple[0],
            width=resolution_tuple[1],
            num_frames=num_frames,
            vae_config=VAE_CONFIGS[vae_name],
        )
        resolution = tuplize(resolution)  # resolution is (height, width)
        self.max_sequence_length = max_sequence_length
        all_tensors = glob.glob(f"{dir_path}/*.pt")
        video_ids = list(set([Path(t).stem.split("_")[-1].split(".")[0] for t in all_tensors]))
        video_ids = sorted([int(i) for i in video_ids])
        self.video_ids = video_ids

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[index]
        vae_latents_path = f"{self.dir_path}/latent_vid_{video_id}.pt"
        text_embeddings_path = f"{self.dir_path}/text_embed_{video_id}.pt"
        vae_latents = torch.load(vae_latents_path, map_location="cpu")
        text_embeddings = torch.load(text_embeddings_path, map_location="cpu")
        return {
            "pixel_values": vae_latents,
            "prompt_embeds": text_embeddings,
        }


def collate_raw_dir_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples into training batch."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # Only convert if needed
    if pixel_values.dtype != torch.float32:
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = [example["prompt_embeds"] for example in examples]
    return_tuple = (pixel_values, prompt_embeds)
    return return_tuple


def setup_data_modules(
    dir_path: str,
    vae_name: str,
    resolution: Union[Tuple[int, int], int],
    max_sequence_length: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
):
    train_dataset = DirDataset(
        dir_path=dir_path,
        vae_name=vae_name,
        resolution=resolution,
        max_sequence_length=max_sequence_length,
        num_frames=num_frames,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_raw_dir_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_dataset, train_dataloader


if __name__ == "__main__":
    train_dataset, train_dataloader = setup_data_modules(
        dir_path="./datasets/3dgs-dissolve/videos",
        vae_name="wan",
        resolution=(480, 832),
        max_sequence_length=512,
        num_frames=81,
        batch_size=1,
        num_workers=1,
    )
    for i in range(len(train_dataset)):
        print(train_dataset[i])
