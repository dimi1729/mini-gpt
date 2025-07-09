import os
import torch
from torch.distributed import init_process_group
from dataclasses import dataclass
import warnings

@dataclass
class DataDDP:
    device: str = "cpu"
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True

    def init_ddp(self) -> bool:
        ddp = int(os.environ.get('RANK', -1)) != -1
        if ddp:
            assert torch.cuda.is_available(), "CUDA is not available"
            init_process_group(backend="nccl")
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
        else:
            ddp_rank = 0
            ddp_local_rank = 0
            ddp_world_size = 1
            master_process = True
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            else:
                warnings.warn("CUDA is not available")

        self.device = device
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process

        return ddp
