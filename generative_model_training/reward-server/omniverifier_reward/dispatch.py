"""Distribute batches evenly across independent, single-GPU verifier replicas."""

from threading import Lock


class RewardPool:
    def __init__(self, workers, submit, collect):
        if not workers:
            raise ValueError("At least one verifier worker is required")
        self.workers = workers
        self.submit = submit
        self.collect = collect
        self._next_worker = 0
        self._lock = Lock()

    def score(self, samples):
        if not samples:
            return []
        with self._lock:
            start = self._next_worker
            self._next_worker = (start + len(samples)) % len(self.workers)
        shards = [[] for _ in self.workers]
        for index, sample in enumerate(samples):
            shards[(start + index) % len(self.workers)].append((index, sample))
        active_shards = [shard for shard in shards if shard]
        tasks = [
            self.submit(worker, [sample for _, sample in shard])
            for worker, shard in zip(self.workers, shards)
            if shard
        ]
        results = self.collect(tasks)
        if len(results) != len(active_shards):
            raise RuntimeError("Verifier returned an incomplete set of shards")
        ordered = [None] * len(samples)
        for shard, scores in zip(active_shards, results):
            if len(scores) != len(shard):
                raise RuntimeError("Verifier returned an incomplete batch")
            for (index, _), score in zip(shard, scores):
                ordered[index] = score
        return ordered
