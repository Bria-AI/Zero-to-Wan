import functools
import gc
import json
import logging
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from models.wan.modules.model import WanAttentionBlock, WanModel
from training.scheduler import FlowMatchScheduler
from training.training_args import get_args
from training.video_dataset import setup_data_modules

# Enable TF32 for faster training
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_device_mesh():
    world_size = dist.get_world_size()
    # dp_replicate = 1 if world_size == 1 else world_size // 8
    dp_replicate = 1
    dp_shard = dist.get_world_size() // dp_replicate

    assert dp_replicate * dp_shard == dist.get_world_size(), (
        f"dp_replicate * dp_shard ({dp_replicate} * {dp_shard}) != world_size ({dist.get_world_size()})"
    )

    dims = []
    names = []
    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    if dp_shard >= 1:
        dims.append(dp_shard)
        names.append("dp_shard")
    dims = tuple(dims)
    names = tuple(names)
    if dist.get_world_size() > 1:
        return init_device_mesh("cuda", mesh_shape=(1, world_size), mesh_dim_names=("dp_replicate", "dp_shard"))
    else:
        return init_device_mesh("cuda", mesh_shape=(1, 1), mesh_dim_names=("dp_replicate", "dp_shard"))


def apply_fsdp_v1(transformer, sharding_strategy, param_dtype, reduce_dtype):
    def arg_to_shard(sharding_strategy):
        if sharding_strategy == "full":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif sharding_strategy == "hybrid_full":
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        elif sharding_strategy == "none":
            sharding_strategy = ShardingStrategy.NO_SHARD
            auto_wrap_policy = None
        elif sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy == "hybrid_zero2":
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        return sharding_strategy

    device_mesh = get_device_mesh()
    fsdp_kwargs = {
        "device_mesh": device_mesh,
        "mixed_precision": MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "sharding_strategy": arg_to_shard(sharding_strategy),
        "device_id": torch.cuda.current_device(),
        "auto_wrap_policy": functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={WanAttentionBlock},
        ),
        "sync_module_states": True,
        "limit_all_gathers": True,
        "use_orig_params": True,
    }

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )
    return transformer


def maybe_drop_prompt(prompt_drop_prob, encoder_hidden_states):
    batch_size = len(encoder_hidden_states)
    random_p = torch.rand(batch_size).to(encoder_hidden_states[0].device)
    prompt_mask = random_p < prompt_drop_prob
    prompt_mask = prompt_mask.reshape(batch_size, 1)
    encoder_hidden_states = [
        encoder_hidden_state * curr_mask for encoder_hidden_state, curr_mask in zip(encoder_hidden_states, prompt_mask)
    ]
    return encoder_hidden_states


def get_transformer(starting_checkpoint_dir: str):
    model = WanModel.from_pretrained(starting_checkpoint_dir)
    return model, model.config


def prepare_extra_input(latents=None):
    return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}


def forward(
    transformer: WanModel,
    scheduler: FlowMatchScheduler,
    batch,
    device,
    prompt_drop_prob,
):
    latents_bcthw, encoder_hidden_states = batch
    latents_bcthw = latents_bcthw.to(device)

    encoder_hidden_states = maybe_drop_prompt(
        prompt_drop_prob=prompt_drop_prob,
        encoder_hidden_states=encoder_hidden_states,
    )

    noise = torch.randn_like(latents_bcthw)
    timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
    timestep = scheduler.timesteps[timestep_id].to(device)
    extra_input = prepare_extra_input(latents_bcthw)
    noisy_latents = scheduler.add_noise(original_samples=latents_bcthw, noise=noise, timestep=timestep)
    training_target = scheduler.training_target(sample=latents_bcthw, noise=noise, timestep=timestep)
    # Compute loss
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(device).type):
        noise_pred = transformer(
            x=noisy_latents,
            t=timestep,
            context=[e[0] for e in encoder_hidden_states],
            seq_len=extra_input["seq_len"],
            use_gradient_checkpointing=True,
        )
        noise_pred_stacked = torch.stack(noise_pred).float()
        loss = torch.nn.functional.mse_loss(noise_pred_stacked.float(), training_target.float())
        loss = loss * scheduler.training_weight(timestep)
    return loss


def avg_scalar_across_ranks(scalar):
    world_size = dist.get_world_size()
    scalar_tensor = torch.tensor(scalar, device="cuda")
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.AVG)
    return scalar_tensor.item()


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(transformer, optimizer, global_step, args, rank):
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
        optim_state = FSDP.optim_state_dict(
            transformer,
            optimizer,
        )

    # todo move to get_state_dict
    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0:
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(transformer.config)
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state, optimizer_path)


def set_debug_env(args):
    """
    FILL YOUR OWN DEBUG PARAMS HERE
    """
    return args


def main():
    args = get_args()
    if args.debug:
        set_debug_env(args)
    # Initialize distributed training
    assert torch.cuda.is_available(), "CUDA is required for training"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # Initialize wandb for the master process
    transformer, model_config = get_transformer(starting_checkpoint_dir=args.starting_checkpoint_dir)

    param_count = sum(p.numel() for p in transformer.parameters())

    if master_process:
        print(f"batch_size: {args.train_batch_size}")
        print(f"model_config: {model_config}")
        print("optimizer_type: adamw")
        print(f"learning_rate: {args.learning_rate}")
        print(f"lr_scheduler_type: {args.lr_scheduler_type}")
        print(f"experiment_name: {args.experiment_name}")
        print(f"param_count: {param_count / 1e6}M")

    if master_process and not args.debug:
        print("USING WANDB")
        if os.environ.get("WANDB_TOKEN"):
            wandb.login(key=os.environ["WANDB_TOKEN"])
        experiment_name = args.experiment_name
        project = "zero_to_wan"
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M")
        wandb.init(project=project, config=args, name=f"{experiment_name}_{timestamp}")
        wandb_enabled = True
    else:
        wandb_enabled = False

    torch.cuda.set_device(device)

    sharding_strategy = args.sharding_strategy
    transformer = apply_fsdp_v1(
        transformer, sharding_strategy.replace("v1_", ""), param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    dist.barrier()

    optimizer = optim.AdamW(
        [param for param in transformer.parameters() if param.requires_grad],
        lr=args.learning_rate,
        betas=(0.95, 0.99),
        fused=True,
    )

    num_warmup_steps = args.num_warmup_steps

    if args.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, args.max_steps)
    elif args.lr_scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, args.max_steps)
    elif args.lr_scheduler_type == "constant":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, 10000000000)
    else:
        raise ValueError(f"Unknown lr scheduler type: {args.lr_scheduler_type}")

    train_dataset, train_dataloader = setup_data_modules(
        dir_path=args.dataset_dir,
        vae_name=args.vae_name,
        resolution=[int(x) for x in args.resolution.split("x")],
        max_sequence_length=args.max_sequence_length,
        num_frames=args.num_frames,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize step counter
    global_step = 0

    # Training loop
    transformer.train()
    start_time_for_log_steps = time.time()

    # clear up memory
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    print(f"Hello from rank={ddp_rank}, world_size={dist.get_world_size()}")

    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    for epoch in range(500000):
        if global_step >= args.max_steps:
            break

        for batch_idx, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            if global_step >= args.max_steps:
                break

            forward_start = time.time()
            diffusion_loss = forward(
                transformer=transformer,
                scheduler=scheduler,
                batch=batch,
                device=device,
                prompt_drop_prob=args.prompt_drop_prob,
            )
            forward_time = time.time() - forward_start
            # Optimization step
            backward_start = time.time()
            optimizer.zero_grad()
            diffusion_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                transformer.parameters(), max_norm=args.grad_clip_norm, foreach=True
            )
            grad_norm = grad_norm.item()
            optimizer.step()
            lr_scheduler.step()
            backward_time = time.time() - backward_start
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            # Logging
            if global_step % args.log_every == 0:
                measured_time_for_log_steps = time.time() - start_time_for_log_steps
                measured_time_for_log_steps = measured_time_for_log_steps / args.log_every

                diffusion_loss_avg = avg_scalar_across_ranks(diffusion_loss.item())

                if master_process:
                    # Calculate average losses per timestep bin
                    avg_fwdbwd_steps = measured_time_for_log_steps
                    print(f"Avg fwdbwd steps: {avg_fwdbwd_steps} sec")
                    # Log metrics to wandb
                    if wandb_enabled:
                        wandb.log(
                            {
                                "train/loss": diffusion_loss_avg,
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                                "train/step": global_step,
                                "train/grad_norm": grad_norm,
                                "timings/step_time": step_time,
                                "timings/backward_time": backward_time,
                                "timings/forward_time": forward_time,
                            },
                            step=global_step,
                        )

                    logger.info(
                        f"Epoch [{epoch}/{500000}] "
                        f"Step [{global_step}/{args.max_steps}] "
                        f"Loss: {diffusion_loss:.4f} "
                        f"Grad Norm: {grad_norm:.6f} "
                        f"Avg fwdbwd steps: {avg_fwdbwd_steps} s "
                        f"LR: {lr_scheduler.get_last_lr()[0]}"
                    )

                start_time_for_log_steps = time.time()

            if global_step % args.checkpoint_every == 0 and global_step > 0 and master_process:
                save_checkpoint(
                    transformer=transformer, optimizer=optimizer, global_step=global_step, args=args, rank=ddp_rank
                )

            global_step += 1

    # Cleanup
    if master_process and wandb_enabled:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
