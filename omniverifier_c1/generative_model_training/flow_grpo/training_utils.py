"""Normalize masked losses across a distributed gradient accumulation window."""


def masked_mean_for_ddp(values, mask, global_valid_count, world_size, accumulation_steps):
    total = values.masked_fill(~mask.bool(), 0).sum()
    if global_valid_count == 0:
        return total * 0
    # DDP averages ranks; the trainer also divides by accumulation_steps later.
    return total * (world_size * accumulation_steps / global_valid_count)
