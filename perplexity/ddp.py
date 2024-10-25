import os
import torch.distributed as dist


def setup_ddp(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()
