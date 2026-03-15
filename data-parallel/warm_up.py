import os

import torch
from torch import distributed


def setup():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device_count = torch.cuda.device_count()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"rank: {rank} | local rank: {local_rank} | world_size: {world_size} | device count: {device_count}")

    distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    t = torch.tensor([rank], device=device, dtype=torch.float32)

    s = torch.arange(8, device=device, dtype=torch.float32)

    gather_slots = [torch.zeros(1, device=device) for _ in range(world_size)]
    distributed.all_gather(gather_slots, t)

    print(f"Device {rank} all-gather: {torch.concat(gather_slots)}")

    scatter_output = torch.zeros(2, device=device)

    distributed.reduce_scatter_tensor(scatter_output, s)

    print(f"Device {rank} reduce-scatter: {scatter_output}")

if __name__ == "__main__":
    setup()
    distributed.destroy_process_group()
