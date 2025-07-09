conda activate mini-gpt
echo $(which python)

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 ./train_mini_gpt.py
