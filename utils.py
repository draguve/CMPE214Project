import os
from typing import Optional

import ray
import ray._private.services as services
import torch.distributed as dist


def clean_ray_init(
    force=None, namespace: Optional[str] = None, dashboard=False
):
    params = {}
    addr = services.canonicalize_bootstrap_address(None)
    if addr is None:
        if force is not None:
            params["num_cpus"] = force
        else:
            params["num_cpus"] = int(os.getenv("RAY_NUM_WORKERS", default=4))

    if namespace is not None:
        params["namespace"] = namespace

    params["include_dashboard"] = (
        int(os.getenv("RAY_DASHBOARD_ENABLE", default=1 if dashboard else 0))
        > 1
    )

    ray.init(**params)


def is_local_server_already_running():
    status = services.canonicalize_bootstrap_address(None)
    return status is not None


def compute_pad_rate(padded_batch, lengths=None, padding_value=0):
    total_tokens = padded_batch.numel()
    if lengths is not None:
        total_valid = lengths.sum().item()
        pad_tokens = total_tokens - total_valid
    else:
        pad_tokens = (padded_batch == padding_value).sum().item()
    pad_rate = pad_tokens / total_tokens
    return pad_rate


def ddp_info():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
    except Exception:
        pass
    return 1, 0
