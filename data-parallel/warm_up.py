import os

import torch


def setup():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device_count = torch.cuda.device_count()

    print(f"rank: {rank} | local rank: {local_rank} | world_size: {world_size} | device count: {device_count}")

if __name__ == "__main__":
    setup()
