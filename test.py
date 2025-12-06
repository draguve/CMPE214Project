import csv
import math
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Literal

import numpy as np
import pynvml
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import typer
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TestTransformer
from raydl import Dataloader as RayLoader
from raydl import get_items_from_queue
from torchdl import RandomSequenceDataset, collate_fn
from utils import clean_ray_init, compute_pad_rate

app = typer.Typer()


def get_gpu_stats(handle):
    """Return (util%, mem_used_MB, mem_total_MB)."""
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    return util.gpu, meminfo.used / 1e6, meminfo.total / 1e6, power


@app.command()
def main(
    results_file: Path = Path("./results.csv"),
    backend: Literal["gloo", "nccl", "mpi"] = "gloo",
    num_workers: int = 4,
    num_sequences: int = 1024,
    batch_size: int = 8,
    vocab_size: int = 1024,
    tqdm_enabled: bool = True,
):
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    if world_size > 1:
        dist.init_process_group(
            backend,
            # init_method=f"file:///{os.getcwd()}/Temp/sharedfile",
            world_size=world_size,
            rank=rank,
        )
        print(
            f"Initialized distributed group: rank {rank}/{world_size}",
            flush=True,
        )
    else:
        print("Running in single-process mode (no distributed init).")

    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    data_to_write = []

    results_file.touch()
    clean_ray_init(force=num_workers * 2, dashboard=False, namespace="testing")
    device = "cuda"

    model = TestTransformer(max_seq_len=1024, vocab_size=1024, d_model=64).to(
        device
    )

    if world_size > 1:
        model = DDP(model, device_ids=[0])
        print(f"[rank {rank}] Model synced", flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters())

    dataset = RandomSequenceDataset(
        num_sequences=num_sequences, vocab_size=vocab_size
    )

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=torch.cuda.is_available(),
        # persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    total_num_sequences = (
        math.ceil(num_sequences / (batch_size * world_size)) + 1
    )

    if world_size > 1:
        dist.barrier()

    total_pad_rate = []
    all_gpu_stats = []
    start_time = time.perf_counter()
    for i, (padded_cpu, lengths) in tqdm(
        enumerate(dl),
        total=total_num_sequences,
        desc="TorchLoader",
        disable=not tqdm_enabled,
    ):
        padded = padded_cpu.to(device, non_blocking=True)
        pad_rate = compute_pad_rate(padded)
        total_pad_rate.append(pad_rate)

        logits = model(padded)  # shape [B, out_dim]
        targets = torch.randn(logits.shape, device=device)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        all_gpu_stats.append(get_gpu_stats(gpu_handle))
    end_time = time.perf_counter()

    total_pad_rate = np.average(np.array(total_pad_rate))
    total_time = end_time - start_time
    all_gpu_stats = np.average(np.array(all_gpu_stats), axis=0)

    new_item = {
        "loader": "TorchDL",
        "prefetch_factor": "",
        "num_seqs": num_sequences,
        "batch_size": batch_size,
        "world_size": world_size,
        "time": total_time,
        "pad_rate": total_pad_rate,
        "occupancy": all_gpu_stats[0],
        "vram_usage": all_gpu_stats[1],
        "total_available_vram": all_gpu_stats[2],
        "power_usage": all_gpu_stats[3],
    }
    data_to_write.append(new_item)
    if rank == 0:
        print(pformat(new_item), flush=True)

    num_sequences_per_node = num_sequences // world_size

    for prefetch_factor in [2, 4, 8, 16, 32]:
        if world_size > 1:
            dist.barrier()

        # RayLoader
        ray_loader = RayLoader.options(
            name="data_loader", lifetime="detached"
        ).remote(
            num_sequnces=int(num_sequences_per_node * 1.5),
            num_generator_workers=num_workers,
            num_collate_workers=2,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            world_size=1,  # this should really be called num_gpus in this case
        )
        queue = ray.get(ray_loader.get_batch_queue.remote(0))
        worker_refs = ray.get(ray_loader.start_round.remote())

        total_pad_rate = []
        all_gpu_stats = []
        start_time = time.perf_counter()
        for index, batch in tqdm(
            enumerate(get_items_from_queue(worker_refs, queue)),
            total=total_num_sequences,
            desc=f"RayLoader {prefetch_factor}",
            disable=not tqdm_enabled,
        ):
            batch_cpu = torch.tensor(batch, dtype=torch.long)
            padded = batch_cpu.to(device, non_blocking=True)
            pad_rate = compute_pad_rate(padded)
            total_pad_rate.append(pad_rate)
            logits = model(padded)  # shape [B, out_dim]
            targets = torch.randn(logits.shape, device=device)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            all_gpu_stats.append(get_gpu_stats(gpu_handle))
            if index >= total_num_sequences:
                break
        end_time = time.perf_counter()

        # empty out queue
        for _ in get_items_from_queue(worker_refs, queue):
            pass

        ray_loader.teardown.remote()
        ray.kill(ray_loader, no_restart=True)

        total_pad_rate = np.average(np.array(total_pad_rate))
        total_time = end_time - start_time
        all_gpu_stats = np.average(np.array(all_gpu_stats), axis=0)
        new_item = {
            "loader": "RayDL",
            "prefetch_factor": prefetch_factor,
            "num_seqs": num_sequences,
            "batch_size": batch_size,
            "world_size": world_size,
            "time": total_time,
            "pad_rate": total_pad_rate,
            "occupancy": all_gpu_stats[0],
            "vram_usage": all_gpu_stats[1],
            "total_available_vram": all_gpu_stats[2],
            "power_usage": all_gpu_stats[3],
        }
        data_to_write.append(new_item)
        if rank == 0:
            print(pformat(new_item), flush=True)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    pynvml.nvmlShutdown()

    if rank == 0:
        write_header = (
            not results_file.exists() or results_file.stat().st_size == 0
        )

        with results_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_to_write[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(data_to_write)


if __name__ == "__main__":
    app()
