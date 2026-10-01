"""Ordered parallel gather into a fixed pinned-buffer ring."""
from concurrent.futures import ThreadPoolExecutor
import bisect

import numpy as np
import torch


class RingLatentLoader:
    def __init__(self, dataset, sampler=None, batch_size=128, device='cuda', workers=2):
        self.dataset = dataset
        self.sampler = sampler if sampler is not None else range(len(dataset))
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.workers = workers
        self.maps = [np.load(dataset.root/s['latents'], mmap_mode='r') for s in dataset.shards]

    def gather(self, indices, slot, copied):
        if copied is not None:
            copied.synchronize()
        x, y = slot
        destination = x.numpy()
        for offset, index in enumerate(indices):
            shard = bisect.bisect_right(self.dataset.ends, index)
            np.copyto(destination[offset], self.maps[shard][index-self.dataset.shards[shard]['start']])
        y.numpy()[:len(indices)] = self.dataset.labels[indices]
        return x[:len(indices)], y[:len(indices)]

    def __iter__(self):
        order = list(self.sampler)
        batches = [order[i:i+self.batch_size] for i in range(0, len(order), self.batch_size)]
        if not batches:
            return
        count = min(self.workers*2, len(batches))
        slots = [(torch.empty((self.batch_size, *self.maps[0].shape[1:]), dtype=torch.float32,
                              pin_memory=True),
                  torch.empty(self.batch_size, dtype=torch.int64, pin_memory=True)) for _ in range(count)]
        stream = torch.cuda.Stream(device=self.device)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.gather, batches[i], slots[i], None) for i in range(count)]
            try:
                for index in range(len(batches)):
                    slot = index % count
                    host_x, host_y = futures[slot].result()
                    with torch.cuda.stream(stream):
                        x = host_x.to(self.device, non_blocking=True)
                        y = host_y.to(self.device, non_blocking=True)
                        copied = stream.record_event()
                    current = torch.cuda.current_stream(self.device)
                    current.wait_event(copied)
                    x.record_stream(current)
                    y.record_stream(current)
                    # A producer may overwrite a pinned slot only after its H2D
                    # completion event. Reuse is independent of GPU compute.
                    if index+count < len(batches):
                        futures[slot] = pool.submit(self.gather, batches[index+count], slots[slot], copied)
                    yield x, y
            finally:
                for future in futures:
                    future.cancel()
                stream.synchronize()
