import html
import re
import time
import warnings
from pathlib import Path

import click
import ftfy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

warnings.filterwarnings("ignore")


def get_text_encoder():
    t5_dtype = torch.bfloat16
    text_model_hf = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="text_encoder").to(
        "cuda", dtype=t5_dtype
    )
    tokenizer_hf = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer")
    return text_model_hf, tokenizer_hf


def encode_prompt(prompt, tokenizer, text_encoder, max_sequence_length=512, device="cuda"):
    dtype = text_encoder.dtype

    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def prompt_clean(text):
        text = whitespace_clean(basic_clean(text))
        return text

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    prompt_embeds = prompt_embeds[:, :seq_lens, :]
    return prompt_embeds


@click.command()
@click.option("--video_dir", type=str, help="Path to the video directory")
@torch.no_grad()
def main(video_dir):
    t0 = time.time()
    text_encoder, tokenizer = get_text_encoder()
    print(f"Time to load text encoder: {time.time() - t0:.2f} seconds")
    video_dir = Path(video_dir)

    prompt_paths = list(video_dir.glob("*.txt"))
    for prompt_path in tqdm(prompt_paths, desc="Processing prompts"):
        with open(prompt_path, "r") as f:
            prompt = f.read()
        text_embed = encode_prompt(prompt=prompt, tokenizer=tokenizer, text_encoder=text_encoder)
        torch.save(text_embed, Path(video_dir) / f"text_embed_{prompt_path.stem}.pt")


if __name__ == "__main__":
    main()
