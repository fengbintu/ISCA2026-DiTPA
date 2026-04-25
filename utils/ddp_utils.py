
import os
import datetime
import torch

def reduce_and_average(data):
    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.AVG)
    return data


def calc_acc_and_reduce(pred, label):
    acc = (pred == label).sum() / (label.numel())
    torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.AVG)
    return acc


def reduce_and_average(data):
    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.AVG)
    return data

def init_distributed_mode(args, cfg, init_method=None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.distributed = True
        args.dist_backend = "nccl"
        torch.cuda.set_device(args.gpu)
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        print(torch.cuda.device_count())
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])

        args.rank = rank
        args.gpu = local_rank
        args.local_rank = local_rank
        args.world_size = world_size
        
        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            import random
            port = random.randint(10000,25000)
            port = 22111

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)
            import time
            time.sleep(3)
        import subprocess
        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_SIZE"] = str(local_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)
        args.distributed = True
        args.dist_backend = "nccl"
        torch.cuda.set_device(local_rank)
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    if not torch.distributed.is_initialized():
        dist_backend = "nccl"
        torch.distributed.init_process_group(
            backend=dist_backend, 
            timeout=datetime.timedelta(seconds=7200),
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )
        torch.distributed.barrier()
        print(torch.distributed.get_world_size())

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        flush = kwargs.pop("flush", True)
        if is_master or force:
            builtin_print(*args, **kwargs, flush=flush)
            

    __builtin__.print = print
