#!/usr/bin/env python
"""Minimal H200 training entrypoint for UniMM-V2X.

This script keeps the UniMM-V2X config, model, dataset, and loss code intact,
but replaces the legacy MMCV runner with an explicit PyTorch training loop that
works in the H200 OpenMMLab 2.x smoke environment.
"""

import argparse
import math
import os
import os.path as osp
import shutil
import subprocess
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction

import h200_compat_smoke  # noqa: F401
import projects.mmdet3d_plugin  # noqa: F401
from mmcv.runner import load_checkpoint
from mmdet3d.registry import MODELS

from projects.mmdet3d_plugin.datasets import custom_build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.unimmv2x.detectors.multi_agent import MultiAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniMM-V2X on H200")
    parser.add_argument("config")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--ignore-missing-load-from", action="store_true")
    parser.add_argument("--resume-from", default=os.environ.get("RESUME_FROM"))
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=os.environ.get("AUTO_RESUME", "0") == "1",
        help="Resume from latest.pth in the work_dir when it exists.",
    )
    parser.add_argument("--base-global-batch", type=int, default=8)
    parser.add_argument("--no-scale-lr", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--hf-path-in-repo", default=os.environ.get("HF_PATH_IN_REPO", "checkpoints"))
    parser.add_argument("--hf-cli", default=os.environ.get("HF_CLI", "/home/gpu_01/miniconda3/envs/mrlora/bin/hf"))
    parser.add_argument(
        "--checkpoint-keep-epochs",
        type=int,
        default=int(os.environ.get("CHECKPOINT_KEEP_EPOCHS", "10")),
        help="Keep the latest N epoch checkpoints and prune older N-epoch blocks.",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_dist():
    if not is_distributed():
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def is_main_process(rank):
    return rank == 0


def set_random_seed(seed, rank):
    seed = int(seed) + int(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_runtime_overrides(cfg, args):
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs_h200", osp.splitext(osp.basename(args.config))[0]
        )

    if args.batch_per_gpu is not None:
        cfg.data.samples_per_gpu = args.batch_per_gpu
        if "train" in cfg.data:
            cfg.data.train.samples_per_gpu = args.batch_per_gpu

    if args.workers_per_gpu is not None:
        cfg.data.workers_per_gpu = args.workers_per_gpu

    if args.max_epochs is not None:
        cfg.total_epochs = args.max_epochs
        if "runner" in cfg:
            cfg.runner.max_epochs = args.max_epochs

    if "optimizer_config" not in cfg:
        cfg.optimizer_config = dict()
    cfg.optimizer_config.cumulative_iters = max(1, int(args.accum))


def apply_lr_scaling(cfg, batch_per_gpu, accum, world_size, base_global_batch, enabled):
    if not enabled:
        return 1.0
    effective_global_batch = int(batch_per_gpu) * int(accum) * int(world_size)
    scale = effective_global_batch / float(base_global_batch)
    cfg.optimizer.lr = cfg.optimizer.lr * scale
    return scale


def build_agent(cfg_model, ignore_missing_load_from=False):
    model = MODELS.build(cfg_model)
    model.init_weights()
    load_from = cfg_model.get("load_from", None)
    if load_from:
        if osp.isfile(load_from):
            load_checkpoint(
                model,
                load_from,
                map_location="cpu",
                revise_keys=[(r"^model_ego_agent\\.", "")],
            )
        elif ignore_missing_load_from:
            print(f"[h200_train] skip missing load_from: {load_from}", flush=True)
        else:
            raise FileNotFoundError(f"Missing checkpoint required by config: {load_from}")
    return model


def build_multi_agent_model(cfg, ignore_missing_load_from=False):
    other_agents = OrderedDict()
    for key in cfg.keys():
        if "model_other_agent" in key:
            other_agents[key] = build_agent(
                cfg.get(key), ignore_missing_load_from=ignore_missing_load_from
            )

    ego_agent = build_agent(
        cfg.model_ego_agent, ignore_missing_load_from=ignore_missing_load_from
    )
    return MultiAgent(ego_agent, other_agents)


def load_top_level_checkpoint(model, cfg, ignore_missing_load_from=False):
    load_from = cfg.get("load_from", None)
    if not load_from:
        return
    if osp.isfile(load_from):
        load_checkpoint(model, load_from, map_location="cpu")
    elif ignore_missing_load_from:
        print(f"[h200_train] skip missing top-level load_from: {load_from}", flush=True)
    else:
        raise FileNotFoundError(f"Missing top-level checkpoint required by config: {load_from}")


def match_paramwise_rule(name, custom_keys):
    matched = None
    for key in sorted(custom_keys.keys(), key=len, reverse=True):
        if key in name:
            matched = custom_keys[key]
            break
    return matched or {}


def build_optimizer(model, optimizer_cfg):
    cfg = optimizer_cfg.copy()
    opt_type = cfg.pop("type")
    paramwise_cfg = cfg.pop("paramwise_cfg", {})
    custom_keys = paramwise_cfg.get("custom_keys", {})
    base_lr = cfg.get("lr")
    base_weight_decay = cfg.get("weight_decay", 0.0)

    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        rule = match_paramwise_rule(name, custom_keys)
        lr_mult = rule.get("lr_mult", 1.0)
        decay_mult = rule.get("decay_mult", 1.0)
        params.append(
            dict(
                params=[param],
                lr=base_lr * lr_mult,
                weight_decay=base_weight_decay * decay_mult,
            )
        )

    cfg["lr"] = base_lr
    cfg["weight_decay"] = base_weight_decay
    if opt_type == "AdamW":
        return torch.optim.AdamW(params, **cfg)
    if opt_type == "Adam":
        return torch.optim.Adam(params, **cfg)
    if opt_type == "SGD":
        return torch.optim.SGD(params, **cfg)
    raise ValueError(f"Unsupported optimizer type for h200_train.py: {opt_type}")


def get_base_lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def optimizer_steps_per_epoch(num_batches, accum):
    return math.ceil(max(0, int(num_batches)) / max(1, int(accum)))


def accumulation_group_size(iter_idx, num_batches, accum):
    accum = max(1, int(accum))
    group_start = (int(iter_idx) // accum) * accum
    return min(accum, int(num_batches) - group_start)


def init_wandb(args, cfg, rank, world_size):
    if not is_main_process(rank) or not args.wandb_project:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[h200_train] wandb import failed: {type(exc).__name__}: {exc}", flush=True)
        return None

    name = args.wandb_name or osp.basename(osp.normpath(cfg.work_dir))
    return wandb.init(
        project=args.wandb_project,
        name=name,
        mode=args.wandb_mode,
        config=dict(
            config_file=args.config,
            work_dir=cfg.work_dir,
            batch_per_gpu=cfg.data.samples_per_gpu,
            accum=cfg.optimizer_config.cumulative_iters,
            world_size=world_size,
            lr=cfg.optimizer.lr,
            total_epochs=cfg.runner.max_epochs if "runner" in cfg else cfg.total_epochs,
        ),
    )


def upload_to_hf(args, work_dir):
    if not args.hf_repo_id:
        print("[h200_train] HF upload skipped: --hf-repo-id is not set", flush=True)
        return
    if not osp.exists(args.hf_cli):
        print(f"[h200_train] HF upload skipped: missing CLI {args.hf_cli}", flush=True)
        return

    subprocess.run(
        [args.hf_cli, "repo", "create", args.hf_repo_id, "--type", "model", "--exist-ok"],
        check=False,
    )
    subprocess.run(
        [
            args.hf_cli,
            "upload",
            args.hf_repo_id,
            work_dir,
            args.hf_path_in_repo,
            "--repo-type",
            "model",
        ],
        check=True,
    )


def update_lr(optimizer, base_lrs, cfg, optimizer_step, total_optimizer_steps):
    lr_cfg = cfg.get("lr_config", {})
    policy = lr_cfg.get("policy", None)
    warmup_iters = int(lr_cfg.get("warmup_iters", 0) or 0)
    warmup_ratio = float(lr_cfg.get("warmup_ratio", 1.0))
    min_lr_ratio = float(lr_cfg.get("min_lr_ratio", 0.0))

    if policy == "CosineAnnealing":
        if total_optimizer_steps <= warmup_iters:
            progress = 1.0
        else:
            progress = (optimizer_step - warmup_iters) / max(
                1, total_optimizer_steps - warmup_iters
            )
            progress = min(max(progress, 0.0), 1.0)
        factor = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        factor = 1.0

    if warmup_iters > 0 and optimizer_step < warmup_iters:
        warmup_progress = optimizer_step / max(1, warmup_iters)
        warmup_factor = warmup_ratio + warmup_progress * (1.0 - warmup_ratio)
        factor *= warmup_factor

    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

    loss = sum(value for key, value in log_vars.items() if "loss" in key)
    log_vars["loss"] = loss
    return loss, log_vars


def move_data_to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    if isinstance(data, dict):
        return {key: move_data_to_cuda(value) for key, value in data.items()}
    if isinstance(data, list):
        return [move_data_to_cuda(value) for value in data]
    if isinstance(data, tuple):
        return tuple(move_data_to_cuda(value) for value in data)
    return data


def make_checkpoint(model, optimizer, cfg, epoch, best_loss=None):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = dict(
        epoch=epoch,
        state_dict=model_to_save.state_dict(),
        optimizer=optimizer.state_dict(),
        config=cfg.pretty_text,
    )
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss
    return checkpoint


def save_checkpoint_atomic(checkpoint, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    if osp.exists(tmp_path):
        os.remove(tmp_path)
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def replace_checkpoint_alias(src, dst):
    tmp_path = f"{dst}.tmp.{os.getpid()}"
    if osp.exists(tmp_path):
        os.remove(tmp_path)
    try:
        try:
            os.link(src, tmp_path)
        except OSError:
            shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def resolve_resume_path(args, cfg):
    if args.resume_from:
        return args.resume_from
    if args.auto_resume:
        latest_path = osp.join(cfg.work_dir, "latest.pth")
        if osp.isfile(latest_path):
            return latest_path
    return None


def move_optimizer_state_to_cuda(optimizer):
    device = torch.device("cuda", torch.cuda.current_device())
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def load_training_state(model, optimizer, resume_from):
    checkpoint = torch.load(resume_from, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(state_dict, strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_cuda(optimizer)
    epoch = int(checkpoint.get("epoch", 0) or 0)
    best_loss = checkpoint.get("best_loss", None)
    if isinstance(best_loss, torch.Tensor):
        best_loss = float(best_loss.detach().cpu())
    return epoch, best_loss


def prune_epoch_checkpoints(work_dir, current_epoch, keep_epochs):
    keep_epochs = int(keep_epochs)
    current_epoch = int(current_epoch)
    if keep_epochs <= 0:
        return []

    old_epoch = current_epoch - keep_epochs
    if old_epoch <= 0:
        return []

    removed = []
    path = osp.join(work_dir, f"epoch_{old_epoch}.pth")
    if osp.exists(path):
        os.remove(path)
        removed.append(path)
    return removed


def reduce_epoch_loss(loss_sum, loss_count):
    if not is_distributed():
        return loss_sum / max(1, loss_count)

    loss_tensor = torch.tensor(
        [float(loss_sum), float(loss_count)],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor[0].item() / max(1.0, loss_tensor[1].item())


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_dist()
    set_random_seed(args.seed, rank)

    cfg = Config.fromfile(args.config)
    apply_runtime_overrides(cfg, args)
    lr_scale = apply_lr_scaling(
        cfg,
        cfg.data.samples_per_gpu,
        cfg.optimizer_config.cumulative_iters,
        world_size,
        args.base_global_batch,
        enabled=not args.no_scale_lr,
    )
    os.makedirs(cfg.work_dir, exist_ok=True)
    if is_main_process(rank):
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        print(f"[h200_train] work_dir={cfg.work_dir}", flush=True)
        print(
            f"[h200_train] batch_per_gpu={cfg.data.samples_per_gpu} "
            f"accum={cfg.optimizer_config.cumulative_iters} world_size={world_size} "
            f"lr={cfg.optimizer.lr:.6g} lr_scale={lr_scale:.4g}",
            flush=True,
        )
    wandb_run = init_wandb(args, cfg, rank, world_size)

    dataset = custom_build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(range(world_size)),
        dist=is_distributed(),
        seed=args.seed,
        shuffler_sampler=cfg.data.get("shuffler_sampler", None),
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )

    model = build_multi_agent_model(
        cfg, ignore_missing_load_from=args.ignore_missing_load_from
    ).cuda()
    load_top_level_checkpoint(
        model, cfg, ignore_missing_load_from=args.ignore_missing_load_from
    )
    if is_distributed():
        find_unused = cfg.get("find_unused_parameters", False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )

    optimizer = build_optimizer(model, cfg.optimizer)
    base_lrs = get_base_lrs(optimizer)
    accum = max(1, int(cfg.optimizer_config.cumulative_iters))
    max_epochs = int(cfg.runner.max_epochs if "runner" in cfg else cfg.total_epochs)
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    optimizer_steps_in_epoch = optimizer_steps_per_epoch(len(data_loader), accum)
    total_optimizer_steps = max_epochs * optimizer_steps_in_epoch
    grad_clip_cfg = cfg.optimizer_config.get("grad_clip", None)
    global_step = 0
    optimizer_step = 0
    best_loss = None
    start_epoch = 0
    resume_from = resolve_resume_path(args, cfg)
    if resume_from:
        if not osp.isfile(resume_from):
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_from}")
        start_epoch, best_loss = load_training_state(model, optimizer, resume_from)
        global_step = start_epoch * max(1, len(data_loader))
        optimizer_step = start_epoch * optimizer_steps_in_epoch
        if is_main_process(rank):
            print(
                f"[h200_train] resumed from {resume_from}; "
                f"start_epoch={start_epoch} global_step={global_step} "
                f"optimizer_step={optimizer_step}",
                flush=True,
            )
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, max_epochs):
        if hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        num_batches_this_epoch = len(data_loader)
        if args.max_iters is not None:
            num_batches_this_epoch = min(
                num_batches_this_epoch, max(0, args.max_iters - global_step)
            )

        for iter_idx, data in enumerate(data_loader):
            if iter_idx >= num_batches_this_epoch:
                break
            update_lr(
                optimizer,
                base_lrs,
                cfg,
                optimizer_step,
                total_optimizer_steps,
            )
            data = move_data_to_cuda(data)
            losses = model(return_loss=True, **data)
            loss, log_vars = parse_losses(losses)
            loss_value = float(log_vars["loss"].detach().cpu())
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            group_size = accumulation_group_size(
                iter_idx, num_batches_this_epoch, accum
            )
            (loss / group_size).backward()

            should_step = (iter_idx + 1) % accum == 0 or (
                iter_idx + 1 == num_batches_this_epoch
            )
            if should_step:
                if grad_clip_cfg is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_cfg.get("max_norm", 0),
                        norm_type=grad_clip_cfg.get("norm_type", 2),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if is_main_process(rank) and global_step % int(cfg.log_config.interval) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[h200_train] epoch={epoch + 1}/{max_epochs} "
                    f"iter={iter_idx + 1}/{len(data_loader)} "
                    f"global_step={global_step} lr={lr:.6g} loss={loss_value:.6f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss_value,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/optimizer_step": optimizer_step,
                        },
                        step=global_step,
                    )

            global_step += 1
            if args.max_iters is not None and global_step >= args.max_iters:
                break

        epoch_loss = reduce_epoch_loss(epoch_loss_sum, epoch_loss_count)
        if is_main_process(rank):
            current_epoch = epoch + 1
            checkpoint_path = osp.join(cfg.work_dir, f"epoch_{current_epoch}.pth")
            latest_path = osp.join(cfg.work_dir, "latest.pth")
            best_path = osp.join(cfg.work_dir, "best.pth")

            is_best = best_loss is None or epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            save_checkpoint_atomic(
                make_checkpoint(model, optimizer, cfg, current_epoch, best_loss),
                checkpoint_path,
            )
            replace_checkpoint_alias(checkpoint_path, latest_path)
            print(
                f"[h200_train] saved {checkpoint_path}; latest={latest_path}",
                flush=True,
            )

            if is_best:
                replace_checkpoint_alias(checkpoint_path, best_path)
                print(
                    f"[h200_train] updated best={best_path} "
                    f"epoch={current_epoch} train_loss={epoch_loss:.6f}",
                    flush=True,
                )

            removed = prune_epoch_checkpoints(
                cfg.work_dir, current_epoch, args.checkpoint_keep_epochs
            )
            if removed:
                print(
                    "[h200_train] pruned old checkpoints: "
                    + ", ".join(osp.basename(path) for path in removed),
                    flush=True,
                )

        if args.max_iters is not None and global_step >= args.max_iters:
            break

    if is_distributed():
        dist.barrier()
    if is_main_process(rank):
        latest_path = osp.join(cfg.work_dir, "latest.pth")
        print(f"[h200_train] final latest checkpoint: {latest_path}", flush=True)
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=osp.basename(osp.normpath(cfg.work_dir)),
                type="model",
            )
            artifact.add_file(latest_path)
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        upload_to_hf(args, cfg.work_dir)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
