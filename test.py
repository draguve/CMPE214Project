import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import typer
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from model import TestTransformer
from torchdl import RandomSequenceDataset, collate_fn
from utils import clean_ray_init, compute_pad_rate

app = typer.Typer()


@app.command()
def main(
    results_file: Annotated[Optional[Path], typer.Option()] = Path(
        "./results.csv"
    ),
    backend: Literal["gloo", "nccl", "mpi"] = "nccl",
    num_workers: int = 4,
    prefect_factor: int = 16,
    num_sequences: int = 1024,
    batch_size: int = 8,
    vocab_size: int = 1024,
):
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    if world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"file:///{os.getcwd()}/Temp/sharedfile",
            world_size=world_size,
            rank=rank,
        )
        print(f"Initialized distributed group: rank {rank}/{world_size}")
    else:
        print("Running in single-process mode (no distributed init).")

    results_file.touch()
    clean_ray_init(force=num_workers * 2, dashboard=False)
    device = "cuda"

    model = TestTransformer(max_seq_len=1024, vocab_size=1024, d_model=64).to(
        device
    )
    if world_size > 1:
        model = DDP(model)

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
    total_num_sequences = math.ceil(num_sequences / (batch_size * world_size))

    total_pad_rate = []
    start_time = time.perf_counter()
    for i, (padded_cpu, lengths) in tqdm(
        enumerate(dl), total=total_num_sequences
    ):
        padded = padded_cpu.to(device, non_blocking=True)
        pad_rate = compute_pad_rate(padded)
        total_pad_rate.append(pad_rate)

        logits = model(padded)  # shape [B, out_dim]
        targets = torch.randn((batch_size, 64), device=device)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    end_time = time.perf_counter()

    total_pad_rate = np.average(np.array(total_pad_rate))
    total_time = end_time - start_time
    print(f"Total pad rate: {total_pad_rate} total_time: {total_time}")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    app()
