"""Deterministic assignments based on the complete input, independent of limit."""
from collections import defaultdict


def shard_assignments(samples, num_shards, strategy="sample"):
    if num_shards < 1 or strategy not in {"sample", "document"}:
        raise ValueError("Invalid shard count or strategy")
    if strategy == "sample":
        return [i % num_shards for i in range(len(samples))]
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample["doc_id"]].append(i)
    # Balance question counts while keeping every PDF on one replica. Using
    # the entire input keeps assignments stable when a smoke limit expands.
    loads = [0] * num_shards
    owners = [None] * len(samples)
    for indices in sorted(groups.values(), key=lambda items: (-len(items), items[0])):
        shard = min(range(num_shards), key=lambda i: (loads[i], i))
        for index in indices:
            owners[index] = shard
        loads[shard] += len(indices)
    return owners


def partition_metadata(num_shards, shard_index, strategy="sample"):
    if strategy == "sample" and num_shards == 1:
        return None
    result = {"num_shards": num_shards, "shard_index": shard_index}
    if strategy != "sample":
        result["strategy"] = "document-v1"
    return result
