import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration for transformer model")

    # Debug mode (used in get_transformer)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced model layers")
    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--resolution", type=str, required=False, default="480x832", help="Resolution of the dataset")
    parser.add_argument(
        "--num_frames", type=int, required=False, default=81, help="Number of frames in videos in the dataset"
    )
    parser.add_argument("--vae_name", type=str, required=False, choices=["wan"], default="wan", help="VAE name")
    parser.add_argument(
        "--max_sequence_length", type=int, required=False, default=512, help="Maximum text sequence length"
    )
    # Model parameters
    parser.add_argument("--starting_checkpoint_dir", type=str, required=True, help="Starting checkpoint directory")
    parser.add_argument(
        "--sharding_strategy",
        choices=["full", "hybrid_full", "none", "shard_grad_op", "hybrid_zero2"],
        default="shard_grad_op",
        help="Sharding strategy",
    )
    # Logging parameters
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    # Training parameters (used in main function)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument(
        "--prompt_drop_prob", type=float, default=0.1, help="Probability of dropping prompts during training"
    )

    parser.add_argument("--gradient_checkpointing", type=str2bool, default=0, help="Use gradient checkpointing")
    # Learning rate scheduler parameters (used in get_lr_scheduler)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        choices=["cosine", "linear", "constant"],
        help="Type of learning rate scheduler",
    )
    parser.add_argument("--num_warmup_steps", type=int, default=10, help="Number of warmup steps for lr scheduler")
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
