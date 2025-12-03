import asyncio
import bisect
import random
from random import randrange, shuffle
from typing import List, Optional

import numpy as np
import ray
import torch
from ray.util.queue import Empty, Queue

from utils import clean_ray_init, compute_pad_rate


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@ray.remote
class TrainItemGeneratorActor:
    def __init__(
        self,
        num_to_generate: int,
        vocab_size: int,
        pre_batch_queue: Queue,
    ):
        self.num_to_generate = num_to_generate
        self.vocab_size = vocab_size
        self.pre_batch_queue = pre_batch_queue

    async def train_item_generator(self):
        for i in range(self.num_to_generate):
            seq_len = random.randint(10, 1000)
            seq = torch.randint(1, self.vocab_size, (seq_len,))
            await self.pre_batch_queue.put_async(
                seq,
                block=True,
                timeout=None,
            )
        return True


def create_batch(tokens, batch_lengths):
    max_size = (
        max(*batch_lengths) if len(batch_lengths) > 1 else batch_lengths[0]
    )
    num_samples = len(batch_lengths)
    batch_tokens = np.zeros((num_samples, max_size), dtype=np.long)
    for i, array_sample in enumerate(tokens):
        batch_tokens[i, : len(array_sample)] = array_sample
    return batch_tokens


async def balanced_add_to_queue(
    post_batch_queues: list[Queue],
    new_batch,
    idx_of_next_queue_to_add_to,
    num_ranks,
):
    await post_batch_queues[idx_of_next_queue_to_add_to].put_async(new_batch)
    idx_of_next_queue_to_add_to += 1
    if idx_of_next_queue_to_add_to >= num_ranks:
        idx_of_next_queue_to_add_to = 0
    return idx_of_next_queue_to_add_to


def create_random_batch(
    prefetch_factor: int,
    batch_size: int,
    lengths: list[int],
    items: list,
):
    start_index = randrange(0, prefetch_factor) * batch_size
    end_index = start_index + batch_size
    batch_items = items[start_index:end_index]
    batch_lengths = lengths[start_index:end_index]
    batch = create_batch(batch_items, batch_lengths)
    lengths = lengths[:start_index] + lengths[end_index:]
    items = items[:start_index] + items[end_index:]
    return lengths, items, batch


def add_to_state(
    target,
    lengths,
    items,
):
    index = bisect.bisect_left(lengths, len(target))
    lengths.insert(index, len(target))
    items.insert(index, target)


@ray.remote
class CollateActor:
    def __init__(
        self,
        prebatch_queue: Queue,
        post_batch_queues: List[Queue],
        prefetch_factor: int,
        batch_size: int,
    ):
        self.prebatch_queue = prebatch_queue
        self.post_batch_queues = post_batch_queues
        self.prefetch_factor = prefetch_factor
        self.batch_size = batch_size

    async def collate_worker(
        self,
        generator_worker_references: List,
    ):
        idx_of_next_queue_to_add_to = 0
        num_ranks = len(self.post_batch_queues)
        # the num_ranks makes we would have items for each rank
        items_to_store = (
            self.prefetch_factor + num_ranks - 1
        ) * self.batch_size

        lengths = []
        items = []

        unfinished_workers = generator_worker_references
        _, unfinished_workers = ray.wait(
            unfinished_workers,
            timeout=0,
            num_returns=len(unfinished_workers),
            fetch_local=False,
        )

        # outer loop keeps going while there are still workers running
        assert self.prebatch_queue.actor is not None
        while unfinished_workers or not await asyncio.gather(
            self.prebatch_queue.actor.empty.remote()
        ):
            # inner loop: pull everything currently in the queue
            while True:
                try:
                    item = await self.prebatch_queue.get_async(
                        block=False, timeout=None
                    )
                except Empty:
                    break

                # process the item
                add_to_state(item, lengths, items)

                # once we've buffered enough, form a batch and enqueue downstream
                if len(lengths) >= items_to_store:
                    lengths, items, batch = create_random_batch(
                        self.prefetch_factor,
                        self.batch_size,
                        lengths,
                        items,
                    )
                    idx_of_next_queue_to_add_to = await balanced_add_to_queue(
                        self.post_batch_queues,
                        batch,
                        idx_of_next_queue_to_add_to,
                        num_ranks,
                    )

            # now that the queue is empty, check which workers are still running
            _, unfinished_workers = ray.wait(
                unfinished_workers,
                timeout=0,
                num_returns=len(unfinished_workers),
                fetch_local=False,
            )

        available_chunks = chunks(list(zip(items, lengths)), self.batch_size)
        shuffle(available_chunks)
        num_to_send_for_balance = (
            (len(available_chunks) - idx_of_next_queue_to_add_to) // num_ranks
        ) * num_ranks

        # add everything from index 0 to the last index to the queue, with sorted batches
        for chunk in available_chunks[:num_to_send_for_balance]:
            batch_items, batch_lengths = zip(*chunk)
            batch = create_batch(batch_items, batch_lengths)
            idx_of_next_queue_to_add_to = await balanced_add_to_queue(
                self.post_batch_queues,
                batch,
                idx_of_next_queue_to_add_to,
                num_ranks,
            )


def empty_ray_queue(q: Queue):
    from queue import Empty

    while True:
        try:
            q.get_nowait()
        except Empty:
            break


# The consumer just needs to check and if the value is None, that means that the training round is finished
@ray.remote
class Dataloader:
    def __init__(
        self,
        num_sequnces: int,
        num_generator_workers: int,
        num_collate_workers: int,
        batch_size: int,
        prefetch_factor=4,
        world_size=1,
    ):
        self.num_sequnces = num_sequnces
        self.num_generator_workers = num_generator_workers
        self.num_collate_workers = num_collate_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.world_size = world_size
        self.num_batches_per_epoch = int(self.get_total_num_batches())
        self.generator_worker_references = None
        self.collate_worker_references = None
        self.prebatch_queue = Queue(
            maxsize=self.prefetch_factor * self.batch_size * self.world_size
        )
        self.output_queues = [
            Queue(maxsize=self.prefetch_factor) for _ in range(self.world_size)
        ]

        assert self.num_sequnces % self.num_generator_workers == 0
        self.generator_actors = [
            TrainItemGeneratorActor.remote(
                self.num_sequnces // self.num_generator_workers,
                1000,
                self.prebatch_queue,
            )
            for _ in range(self.num_generator_workers)
        ]
        self.collate_actors = [
            CollateActor.remote(
                self.prebatch_queue,
                self.output_queues,
                self.prefetch_factor,
                self.batch_size,
            )
            for _ in range(self.num_collate_workers)
        ]

    def get_total_num_batches(self):
        return self.num_sequnces // self.batch_size

    def get_batch_queue(self, rank) -> Optional[Queue]:
        if self.output_queues is None:
            return None
        return self.output_queues[rank]

    # setups up pre batch queue to be fed by generators
    # setups up the output queue's to be fed by the collate_workers (one for each rank)
    def start_round(self):
        # check if previous still running
        if self.generator_worker_references is not None:
            _, running = ray.wait(
                self.generator_worker_references,
                timeout=0,
                num_returns=len(self.generator_worker_references),
            )
            if running:
                raise AssertionError("Previous generators still running")

        if self.collate_worker_references is not None:
            _, running = ray.wait(
                self.collate_worker_references,
                timeout=0,
                num_returns=len(self.collate_worker_references),
            )
            if running:
                raise AssertionError("Previous collate worker still running")

        # raise error if queues are not empty
        assert self.prebatch_queue.empty()
        for queue in self.output_queues:
            assert queue.empty()

        self.generator_worker_references = [
            actor.train_item_generator.remote()
            for actor in self.generator_actors
        ]
        self.collate_worker_references = [
            actor.collate_worker.remote(self.generator_worker_references)
            for actor in self.collate_actors
        ]
        return self.collate_worker_references

    def get_collate_worker_references(self):
        return self.collate_worker_references

    def stop_queue(self):
        if self.generator_worker_references is not None:
            for ref in self.generator_worker_references:
                ray.cancel(ref)
        if self.collate_worker_references is not None:
            for ref in self.collate_worker_references:
                ray.cancel(ref)

        empty_ray_queue(self.prebatch_queue)
        for queue in self.output_queues:
            empty_ray_queue(queue)

        assert self.prebatch_queue.empty()
        for queue in self.output_queues:
            assert queue.empty()

        self.generator_worker_references = None
        self.collate_worker_references = None
        print("Queue Force Stopped")


def get_items_from_queue(
    collate_worker_references,
    queue,
):
    unfinished_workers = collate_worker_references
    _, unfinished_workers = ray.wait(
        unfinished_workers,
        timeout=0,
        num_returns=len(unfinished_workers),
        fetch_local=False,
    )

    while unfinished_workers or not queue.empty():
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            yield item
        _, unfinished_workers = ray.wait(
            unfinished_workers,
            timeout=0,
            num_returns=len(unfinished_workers),
            fetch_local=False,
        )


def test():
    clean_ray_init(namespace="test")
    loader = Dataloader.options(name="data_loader", lifetime="detached").remote(
        num_sequnces=1000,
        num_generator_workers=4,
        num_collate_workers=2,
        batch_size=8,
        prefetch_factor=16,
        world_size=1,
    )
    queue = ray.get(loader.get_batch_queue.remote(0))
    worker_refs = ray.get(loader.start_round.remote())
    for batch in get_items_from_queue(worker_refs, queue):
        batch = torch.tensor(batch, dtype=torch.long)
        print(f"Pad rate {compute_pad_rate(batch)}")
        print("Batch shape:", batch.shape)
        break


if __name__ == "__main__":
    test()
