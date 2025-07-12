import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import time

from config import Config
from dataloader import DataLoader
from learning_rate_scheduler import LearningRateScheduler
from distributed_data_processing import DataDDP
from model import MiniGPT

if __name__ == '__main__':

    ddp = DataDDP()
    use_ddp = ddp.init_ddp()

    if ddp.master_process:
        wandb.init(
            project="mini-gpt",
            name="mini-shakespeare",
            config={
                "batch_size": 32768
            }
        )

    # Use gpt3 ~0.5M batch size with grad accumulation
    debug = True
    if debug:
        total_batch_size = 32768  # 2**16
    else:
        total_batch_size = 524288  # 2**19
    B = 4
    T = 512
    assert total_batch_size % (B * T) == 0, "Batch size must be divisible by B"
    grad_accum_steps = total_batch_size // (B * T)

    dataloader = DataLoader(B=B, T=T, process_rank=ddp.ddp_local_rank, num_processes=ddp.ddp_world_size, device=ddp.device)

    torch.set_float32_matmul_precision('high') # TF32 precision

    config = Config(vocab_size=50304) # 50304 is div by 128 so it is a "nicer number" than 50257
    model = MiniGPT(config)
    model.to(ddp.device)
    model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[ddp.ddp_local_rank], output_device=ddp.device)
    # logits, loss = model(x, y)
    # print(loss)

    initial_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    lr_scheduler = LearningRateScheduler(initial_optimizer, warmup_steps=100, max_steps=10000)

    if ddp.master_process:
        wandb.watch(model, log="all", log_freq=10)

    for i in range(10000):
        t0 = time.time()
        lr_scheduler.optimizer.zero_grad()

        total_loss = 0.0
        for sub_step in range(grad_accum_steps):
            x, y = dataloader.next_batch()
            with torch.autocast(device_type=ddp.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            if use_ddp:
                model.require_backward_grad_sync = (sub_step == grad_accum_steps - 1)
            loss.backward()
        if use_ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr_scheduler.set_lr(i)
        lr_scheduler.optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_s = (dataloader.B * dataloader.T) * ddp.ddp_world_size / dt
        if ddp.master_process:
            print(f"Step {i+1}: Loss = {total_loss}, time = {dt*1000:.2f} ms, tok/s = {tokens_per_s:.2f}, norm = {norm:.3f}, lr = {lr_scheduler.get_lr(i):.5f}")
            wandb.log({
                "train/loss": total_loss,
                "train/tokens_per_second": tokens_per_s,
                "train/norm": norm,
                "train/lr": lr_scheduler.get_lr(i)
            }, step=i+1)

            if i > 0 and i % 1000 == 0:
                checkpoint_path = f"checkpoints/checkpoint_{i}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

    if use_ddp:
        dist.destroy_process_group()
