import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt_location = Path("Plots")

os.makedirs(plt_location, exist_ok=True)

df = pd.read_csv("final_ext.csv")
df["effective_batch_size"] = df["batch_size"] * df["world_size"]
df["total_ram_usage"] = df["vram_usage"] * df["world_size"]
df["throughput"] = df["num_seqs"] / df["time"]
df["prefetch_factor"] = df["prefetch_factor"].fillna(0).astype(int)

print(df)

df_single = df.loc[(df["world_size"] == 1) & (df["batch_size"] == 256)]

plt.figure(figsize=(10, 6))
plt.plot(
    df_single["prefetch_factor"],
    df_single["occupancy"],
    marker="o",
    linewidth=2,
)
plt.xlabel("prefetch_factor (0 = TorchDL)")
plt.title("Occupancy vs Prefetch Factor (Batch Size = 256, World Size=1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plt_location / "OccVsPrefetchFactor_B256_W1.svg", dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(
    df_single["prefetch_factor"],
    df_single["pad_rate"],
    marker="o",
    linewidth=2,
)
plt.xlabel("prefetch_factor (0 = TorchDL)")
plt.title("Pad Rate vs Prefetch Factor (Batch Size = 256, World Size=1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plt_location / "PadRateVsPrefetchFactor_B256_W1.svg", dpi=300)

world_sizes = [1, 2, 4, 8]

for w in world_sizes:
    df_w = df[df["world_size"] == w].copy()

    # TorchDL has no prefetch_factor → treat as 0
    df_w["pf_plot"] = df_w["prefetch_factor"].fillna(0)

    # === OCCUPANCY vs PREFETCH for all batch sizes ===
    plt.figure(figsize=(10, 6))
    for bs, sub in df_w.groupby("batch_size"):
        sub = sub.sort_values("pf_plot")
        plt.plot(
            sub["pf_plot"],
            sub["occupancy"],
            marker="o",
            linewidth=2,
            label=f"batch={int(bs)}",
        )

    plt.xlabel("prefetch_factor (0 = TorchDL)")
    plt.ylabel("Occupancy (%)")
    plt.title(f"Occupancy vs Prefetch Factor (All Batch Sizes, World Size={w})")
    plt.legend(title="Batch size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plt_location / f"OccVsPrefetchFactor_AllB_W{w}.svg",
        dpi=300,
    )

    # === PAD RATE vs PREFETCH for all batch sizes ===
    plt.figure(figsize=(10, 6))
    for bs, sub in df_w.groupby("batch_size"):
        sub = sub.sort_values("pf_plot")
        plt.plot(
            sub["pf_plot"],
            sub["pad_rate"],
            marker="o",
            linewidth=2,
            label=f"batch={int(bs)}",
        )

    plt.xlabel("prefetch_factor (0 = TorchDL)")
    plt.ylabel("Pad Rate")
    plt.title(f"Pad Rate vs Prefetch Factor (All Batch Sizes, World Size={w})")
    plt.legend(title="Batch size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plt_location / f"PadRateVsPrefetchFactor_AllB_W{w}.svg",
        dpi=300,
    )


ray = df[df["loader"] == "RayDL"].copy()
torchdl = df[df["loader"] == "TorchDL"].copy()

for world_size_of_interest in [1, 2, 4, 8]:
    # Filter RayDL and TorchDL for the desired world_size
    ray_ws = ray[ray["world_size"] == world_size_of_interest]
    torch_ws = torchdl[torchdl["world_size"] == world_size_of_interest]

    plt.figure(figsize=(8, 5))

    # --- RayDL lines (multiple based on prefetch_factor) ---
    for pf, sub in ray_ws.groupby("prefetch_factor"):
        sub = sub.sort_values("batch_size")
        label = f"RayDL prefetch={int(pf)}"
        plt.plot(
            sub["batch_size"],
            sub["throughput"],
            marker="o",
            label=label,
        )

    # --- Add TorchDL line ---
    if len(torch_ws) > 0:
        torch_ws = torch_ws.sort_values("batch_size")
        plt.plot(
            torch_ws["batch_size"],
            torch_ws["throughput"],
            marker="o",
            linestyle="--",
            linewidth=2,
            label="TorchDL",
        )

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (seq/s)")
    plt.title(f"Throughput vs Batch Size (world_size={world_size_of_interest})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plt_location / f"ThroughputvsBatchSize_W{world_size_of_interest}.svg",
        dpi=300,
    )

for world_size_of_interest in [1, 2, 4, 8]:
    # Filter RayDL and TorchDL for the desired world_size
    ray_ws = ray[ray["world_size"] == world_size_of_interest]
    torch_ws = torchdl[torchdl["world_size"] == world_size_of_interest]

    plt.figure(figsize=(8, 5))

    # --- RayDL lines (multiple based on prefetch_factor) ---
    for pf, sub in ray_ws.groupby("prefetch_factor"):
        sub = sub.sort_values("batch_size")
        label = f"RayDL prefetch={int(pf)}"
        plt.plot(
            sub["batch_size"],
            sub["time"],
            marker="o",
            label=label,
        )

    # --- Add TorchDL line ---
    if len(torch_ws) > 0:
        torch_ws = torch_ws.sort_values("batch_size")
        plt.plot(
            torch_ws["batch_size"],
            torch_ws["time"],
            marker="o",
            linestyle="--",
            linewidth=2,
            label="TorchDL",
        )

    plt.xlabel("Batch size")
    plt.ylabel("Time (s)")
    plt.title(f"Time vs Batch Size (world_size={world_size_of_interest})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plt_location / f"TimetvsBatchSize_W{world_size_of_interest}.svg",
        dpi=300,
    )

    # 5. Power Usage vs. Throughput (compute efficiency)
    plt.figure(figsize=(8, 5))

    # RayDL points (grouped by prefetch factor)
    for pf, sub in ray_ws.groupby("prefetch_factor"):
        plt.scatter(
            sub["throughput"],
            sub["power_usage"],
            label=f"RayDL prefetch={int(pf)}",
            alpha=0.8,
        )

    # TorchDL points (single group)
    if len(torch_ws) > 0:
        plt.scatter(
            torch_ws["throughput"],
            torch_ws["power_usage"],
            label="TorchDL",
            marker="x",
            s=80,
        )

    plt.xlabel("Throughput (seq/s)")
    plt.ylabel("Power usage (W)")
    plt.title(
        f"Power Usage vs Throughput (world_size={world_size_of_interest})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plt_location
        / f"PowerUsage_vs_Throughput_W{world_size_of_interest}.svg",
        dpi=300,
    )

plt.figure(figsize=(8, 6))

# TorchDL (prefetch_factor = NaN) → plot separately
torchdl = df[df["loader"] == "TorchDL"]
if len(torchdl) > 0:
    plt.scatter(
        torchdl["pad_rate"],
        torchdl["occupancy"],
        label="TorchDL (pf=0)",
        marker="x",
        s=80,
        linewidths=2,
    )

# RayDL points colored/labeled by prefetch factor
ray = df[df["loader"] == "RayDL"]
for pf, sub in ray.groupby("prefetch_factor"):
    plt.scatter(
        sub["pad_rate"],
        sub["occupancy"],
        label=f"prefetch={int(pf)}",
        alpha=0.8,
    )

plt.xlabel("Padding Rate")
plt.ylabel("Occupancy (%)")
plt.title("Occupancy vs Padding Rate")
plt.grid(True, alpha=0.3)
plt.legend(title="Loader / Prefetch")
plt.tight_layout()

plt.savefig(
    plt_location / "Occupancy_vs_PaddingRate.svg",
    dpi=300,
)

batch_sizes_of_interest = [32, 64, 128, 256]

for batch_size_of_interest in batch_sizes_of_interest:
    # Filter RayDL and TorchDL for the desired batch_size
    ray_bs = ray[ray["batch_size"] == batch_size_of_interest]
    torch_bs = torchdl[torchdl["batch_size"] == batch_size_of_interest]

    plt.figure(figsize=(8, 5))

    # --- RayDL lines (multiple based on prefetch_factor) ---
    for pf, sub in ray_bs.groupby("prefetch_factor"):
        sub = sub.sort_values("world_size")
        label = f"RayDL prefetch={int(pf)}"
        plt.plot(
            sub["world_size"],
            sub["throughput"],
            marker="o",
            label=label,
        )

    # --- Add TorchDL line ---
    if len(torch_bs) > 0:
        torch_bs_sorted = torch_bs.sort_values("world_size")
        plt.plot(
            torch_bs_sorted["world_size"],
            torch_bs_sorted["throughput"],
            marker="o",
            linestyle="--",
            linewidth=2,
            label="TorchDL",
        )

    plt.xlabel("World size")
    plt.ylabel("Throughput (seq/s)")
    plt.title(f"Throughput vs World Size (batch_size={batch_size_of_interest})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plt_location / f"ThroughputvsWorldSize_B{batch_size_of_interest}.svg",
        dpi=300,
    )


# New Plot
longer_run_df = pd.read_csv("final_results2.csv")
longer_run_df["effective_batch_size"] = (
    longer_run_df["batch_size"] * longer_run_df["world_size"]
)
longer_run_df["total_ram_usage"] = (
    longer_run_df["vram_usage"] * longer_run_df["world_size"]
)
longer_run_df["throughput"] = longer_run_df["num_seqs"] / longer_run_df["time"]
longer_run_df["prefetch_factor"] = (
    longer_run_df["prefetch_factor"].fillna(0).astype(int)
)

world_size_of_interest = 8
ray_ws = longer_run_df[longer_run_df["loader"] == "RayDL"].copy()
torch_ws = longer_run_df[longer_run_df["loader"] == "TorchDL"].copy()

plt.figure(figsize=(8, 5))

# --- RayDL lines (multiple based on prefetch_factor) ---
for pf, sub in ray_ws.groupby("prefetch_factor"):
    sub = sub.sort_values("batch_size")
    label = f"RayDL prefetch={int(pf)}"
    plt.plot(
        sub["batch_size"],
        sub["time"],
        marker="o",
        label=label,
    )

# --- Add TorchDL line ---
if len(torch_ws) > 0:
    torch_ws = torch_ws.sort_values("batch_size")
    plt.plot(
        torch_ws["batch_size"],
        torch_ws["time"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="TorchDL",
    )

plt.xlabel("Batch size")
plt.ylabel("Time (s)")
plt.title(f"Time vs Batch Size (world_size={world_size_of_interest})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plt_location / f"TimetvsBatchSize_W{world_size_of_interest}_LongerRun.svg",
    dpi=300,
)

plt.figure(figsize=(8, 5))

# --- RayDL lines (multiple based on prefetch_factor) ---
for pf, sub in ray_ws.groupby("prefetch_factor"):
    sub = sub.sort_values("batch_size")
    label = f"RayDL prefetch={int(pf)}"
    plt.plot(
        sub["batch_size"],
        sub["throughput"],
        marker="o",
        label=label,
    )

# --- Add TorchDL line ---
if len(torch_ws) > 0:
    torch_ws = torch_ws.sort_values("batch_size")
    plt.plot(
        torch_ws["batch_size"],
        torch_ws["throughput"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="TorchDL",
    )

plt.xlabel("Batch size")
plt.ylabel("Throughput (seq/s)")
plt.title(f"Throughput vs Batch Size (world_size={world_size_of_interest})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plt_location
    / f"ThroughputvsBatchSize_W{world_size_of_interest}_LongerRun.svg",
    dpi=300,
)

# 5. Power Usage vs. Throughput (compute efficiency)
plt.figure(figsize=(8, 5))

# RayDL points (grouped by prefetch factor)
for pf, sub in ray_ws.groupby("prefetch_factor"):
    plt.scatter(
        sub["throughput"],
        sub["power_usage"],
        label=f"RayDL prefetch={int(pf)}",
        alpha=0.8,
    )

# TorchDL points (single group)
if len(torch_ws) > 0:
    plt.scatter(
        torch_ws["throughput"],
        torch_ws["power_usage"],
        label="TorchDL",
        marker="x",
        s=80,
    )

plt.xlabel("Throughput (seq/s)")
plt.ylabel("Power usage (W)")
plt.title(f"Power Usage vs Throughput (world_size={world_size_of_interest})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    plt_location / f"PowerUsage_vs_Throughput_W{world_size_of_interest}.svg",
    dpi=300,
)
