from pathlib import Path

import click
import numpy as np
import torch
import torchvision.transforms.functional as TF
import decord

from tqdm import tqdm

from models.wan.modules.clip import CLIPModel
from models.wan.modules.vae import WanVAE


def get_vae_model(checkpoint_dir: Path):
    vae = WanVAE(vae_pth=checkpoint_dir / "Wan2.1_VAE.pth", device="cuda")
    return vae


def get_clip_model(checkpoint_dir: Path):
    clip_dtype = torch.float16
    clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    clip_tokenizer = "xlm-roberta-large"
    device = torch.device("cuda")
    clip = CLIPModel(
        dtype=clip_dtype,
        device=device,
        checkpoint_path=checkpoint_dir / clip_checkpoint,
        tokenizer_path=checkpoint_dir / clip_tokenizer,
    )
    return clip


@click.command()
@click.option("--video_dir", type=str, help="Path to the video directory")
@click.option("--target_fps", type=float, default=16.0, help="Target fps for the video")
@click.option("--num_frames", type=int, default=81, help="Number of frames to sample")
@click.option("--target_height", type=int, default=480, help="Target height for the video")
@click.option("--target_width", type=int, default=832, help="Target width for the video")
@click.option("--compute_clip_context", type=bool, default=False, help="Whether to compute clip context")
@click.option("--vae_checkpoint_dir", type=str, help="Path to the model checkpoint directory")
@click.option("--clip_checkpoint_dir", type=str, help="Path to the model checkpoint directory")
@torch.no_grad()
def main(
    video_dir,
    target_fps,
    num_frames,
    target_height,
    target_width,
    compute_clip_context,
    vae_checkpoint_dir,
    clip_checkpoint_dir,
):
    video_dir = Path(video_dir)
    output_dir = video_dir
    target_video_len = (num_frames - 1) / target_fps
    vae = get_vae_model(Path(vae_checkpoint_dir))
    if compute_clip_context:
        clip = get_clip_model(Path(clip_checkpoint_dir))

    video_paths = sorted(list(video_dir.glob("*.mp4")))
    for video_path in tqdm(video_paths, desc="Processing videos"):
        print(f"processing {video_path}")
        latents_output_path = Path(output_dir) / f"latent_vid_{video_path.stem}.pt"
        if latents_output_path.exists():
            print(f"latents already exist for {video_path}, skipping")
            continue
        video_reader = decord.VideoReader(str(video_path), height=target_height, width=target_width)
        vid_fps = video_reader.get_avg_fps()
        num_original_frames = min(len(video_reader), int(vid_fps * target_video_len))
        sampled_frames_from_clip_len_in_secs = np.linspace(
            0, num_original_frames-1, min(num_original_frames, num_frames), endpoint=True, dtype=int
        )
        video = video_reader.get_batch(sampled_frames_from_clip_len_in_secs)
        video = video.asnumpy()

        vae_input_vid = torch.stack([TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda") for img in video], dim=0)
        latent_vid = vae.encode(vae_input_vid.permute(1, 0, 2, 3).unsqueeze(0))[0]
        torch.save(latent_vid, latents_output_path)
        if compute_clip_context:
            clip_context = clip.visual(vae_input_vid[0][:, None, :, :].unsqueeze(0))
            torch.save(clip_context, Path(output_dir) / f"clip_context_{video_path.stem}.pt")


if __name__ == "__main__":
    main()
