import os
import time

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

    print(f"Device {rank} all-gather: {torch.concat(gather_slots)}\n")

    scatter_output = torch.zeros(2, device=device)

    distributed.reduce_scatter_tensor(scatter_output, s)

    print(f"Device {rank} reduce-scatter: {scatter_output}\n")

    u = torch.arange(8, device=device, dtype=torch.float32)

    if rank == 0:
        print(f"Before all-reduce: {u}")
        torch.cuda.synchronize(device)
        start = time.time()
    distributed.all_reduce(u, op=distributed.ReduceOp.SUM)
    if rank == 0:
        print(f"After all-reduce: {u}")
        torch.cuda.synchronize()
        delta = time.time() - start
        print(f"Time for all-reduce: {delta}")
    
    u = torch.arange(8, device=device, dtype=torch.float32)
    scatter_output = torch.zeros(2, device=device)
    gather_slots = [torch.zeros(2, device=device) for _ in range(world_size)]

    if rank == 0:
        torch.cuda.synchronize()
        start = time.time()
    distributed.reduce_scatter_tensor(scatter_output, u, op=distributed.ReduceOp.SUM)
    distributed.all_gather(gather_slots, scatter_output)
    if rank == 0:
        torch.cuda.synchronize()
        delta = time.time() - start
        print(f"Time for reduce-scatter + all-gather: {delta}")
    print(f"reduce-Scatter + all-gather: {torch.concat(gather_slots)}")





if __name__ == "__main__":
    setup()
    distributed.destroy_process_group()
