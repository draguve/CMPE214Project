import random

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from utils import compute_pad_rate, ddp_info


class RandomSequenceDataset(IterableDataset):
    def __init__(
        self, num_sequences: int, vocab_size: int = 1000, seed: int = 42
    ):
        super().__init__()
        self.num_sequences = int(num_sequences)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

    def __iter__(self):
        world_size, rank = ddp_info()

        worker = get_worker_info()
        if worker is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker.num_workers
            worker_id = worker.id

        total_equal = (self.num_sequences // world_size) * world_size
        per_rank = total_equal // world_size

        rank_start = rank * per_rank
        rank_end = rank_start + per_rank  # exclusive

        base = per_rank // num_workers
        rem = per_rank % num_workers
        extra = 1 if worker_id < rem else 0
        prev_extras = min(worker_id, rem)
        w_count = base + extra
        w_offset = worker_id * base + prev_extras

        w_start = rank_start + w_offset
        w_end = w_start + w_count  # exclusive

        if w_start >= w_end or per_rank == 0:
            return iter(())

        for _ in range(w_start, w_end):
            seq_len = random.randint(10, 1000)
            seq = torch.randint(1, self.vocab_size, (seq_len,))
            yield seq


def collate_fn(batch):
    lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0
    )
    return padded_batch, lengths


if __name__ == "__main__":
    dataset = RandomSequenceDataset(
        num_sequences=1000, vocab_size=5000, seed=42
    )
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    for padded_batch, lengths in dataloader:
        print(f"Pad rate {compute_pad_rate(padded_batch)}")
        print("Batch shape:", padded_batch.shape)
        break
