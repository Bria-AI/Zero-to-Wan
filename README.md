# Zero-to-Wan
A minimalistic repo to finetune Wan2.1-1.3B


<details>
  <summary><strong>Demo Videos</strong></summary>

  <div align="center">
  <video src="https://github.com/user-attachments/assets/395b9f48-8335-41ff-ae79-6b25c6b1cf4c">
  </video> </div>

  <div align="center">
  <video src="https://github.com/user-attachments/assets/146b0951-860f-44c9-8f7f-5e4ec3a14c98">
  </video></div>
  
  <div align="center">
  <video src="https://github.com/user-attachments/assets/edcb516b-de6f-48b9-95b3-c9bf4434b4c7">
  </video></div>
  
  <div align="center">
  <video src="https://github.com/user-attachments/assets/8ba7d37e-1d13-4d29-a479-696be1504816">
  </video></div>

</details>





## Introduction
This is a minimalistic, hackable repo to finetune Wan2.1-1.3B on some simple effects courtesy of [Hugging Face](https://huggingface.co/datasets/finetrainers/3dgs-dissolve).
The repo includes the implementation of the [Wan Model](https://github.com/Wan-Video/Wan2.1/tree/main) and its finetuning using plain PyTorch.

We'd like to also thank the authors of [DiffSynth](https://github.com/modelscope/DiffSynth-Studio/tree/main), [FastVideo](https://github.com/hao-ai-lab/FastVideo/) for their great work which we build on in this repo.

The stages of training implemented in this repo are:
1. Data preparation
    - Downloading the 3DGS-Dissolve dataset from Hugging Face
    - Captioning the videos
    - Precomputing text embeddings for the captions
    - Precomputing video latents using the VAE for the videos
2. Training
3. Generation using a newly trained checkpoint

## Setup
1. Clone the repo
```
git clone https://github.com/Bria-AI/Zero-to-Wan
cd Zero-to-Wan
```

2. Install `uv`:
   
Instructions taken from [here](https://docs.astral.sh/uv/getting-started/installation/).

For linux systems this should be:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install the dependencies:
```
uv sync --no-install-package flash-attn
uv sync
```
4. Activate your `.venv` and set the Python env:
```
source .venv/bin/activate
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

5. Run the data preparation script
```
chmod +x scripts/run_data_prep.sh
./scripts/run_data_prep.sh
```

6. Run the training script
```
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

7. Run the generation script
currently the generation script uses checkpoint-1000 as the finetuned checkpoint, you can change it to any other checkpoint you want to use for generation.
```
chmod +x scripts/run_generate.sh
./scripts/run_generate.sh
```

