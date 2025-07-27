export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch


torchrun --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${MASTER_PORT} internvl/train/internvl_chat_pretrain.py \
  --conv_style "internlm2-chat" \
  --output_dir /path/to/save \
  --meta_path /path/to/data \
  --vision_path /path/to/InternViT \
  --llm_path /path/to/InternLM \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "ds_configs/zero_stage1_config.json" \
  --report_to "tensorboard" \
  --use_mgm True \
  --use_custom_trainer True \