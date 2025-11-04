import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

import torch
import wandb
import time
import glob
import json
import torch.distributed as dist
from transformers import set_seed
set_seed(0)

import torch._dynamo
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.capture_scalar_outputs = True


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

PARQUET_DIR = os.path.join(OUTPUT_DIR, "parquets_1")

TRAIN_STRATEGY = 'base'

# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 16,
    'save_steps': 10000,
    'save_total_limit': 200,
    'learning_rate': 4e-4,
    'weight_decay': 0.1,
    'warmup_steps': 100,
    'logging_steps': 100,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    # 'gradient_checkpointing': True,
    'gradient_accumulation_steps': 2,
    'dataloader_num_workers': 8,
    # 'dataloader_pin_memory': True,
    'torch_compile': False,
    'report_to': 'wandb',
    'bf16': True,
    'tf32': True,
}


class TimeoutCallback(TrainerCallback):
    def __init__(self, timeout_seconds, check_every_n_steps=1):
        self.timeout_seconds = float(timeout_seconds)
        self.check_every_n_steps = int(check_every_n_steps)
        self.start_time = None
        self.step = 0
        self.is_dist = False
        self.rank = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()
        self.is_dist = dist.is_available() and dist.is_initialized()
        if self.is_dist:
            self.rank = dist.get_rank()

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.check_every_n_steps != 0:
            return control

        local_stop = False
        if self.rank == 0:
            local_stop = (time.monotonic() - self.start_time) > self.timeout_seconds

        if self.is_dist:
            flag = [local_stop]
            dist.broadcast_object_list(flag, src=0)
            should_stop = bool(flag[0])
        else:
            should_stop = local_stop

        if should_stop:
            control.should_training_stop = True
            if (not self.is_dist or self.rank == 0) :
                elapsed = time.monotonic() - self.start_time
                print(f"Training stopped after {elapsed:.2f}s (timeout)")
        return control


def prepare_tokenizer():
    """
    TODO: Implement tokenizer preparation.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    TODO: Implement tokenization function.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    texts = examples.get('text', None)

    tokenzr = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )
    
    labels = []
    for ids, attn in zip(tokenzr[INPUT_IDS], tokenzr[ATTENTION_MASK]):
        labels.append([(idd if att == 1 else -100) for idd, att in zip(ids, attn)])

    tokenzr[LABELS] = labels
    
    return tokenzr


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    TODO: Implement saving dataset as parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=idx)
        shard.to_parquet(os.path.join(output_dir, f"{idx}.parquet"))


def prepare_dataset():
    """
    TODO: Implement dataset preparation.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")

    tokenizer = prepare_tokenizer()
    tokenized = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=['id', 'url', 'title'],
        desc="Tokenizing",
    )

    save_as_parquets(tokenized, output_dir=PARQUET_DIR, num_shards=NUM_SHARDS)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    TODO: Implement loading of tokenized dataset from parquet files.
    - List all parquet files in data_dir
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    ds = load_dataset('parquet', data_files=files, split='train')
    return ds


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model

def initialize_wandb():
    os.environ['USE_WANDB']='1'
    if os.getenv('RANK', '0') == '0':
        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'llm_pretrain_hw2'),
            name=os.getenv('WANDB_RUN_NAME', 'run_1gpu_version_for_compare'),
            settings=wandb.Settings(
                http_proxy=os.getenv('AVITO_HTTP_PROXY','http://prx-squid-rev.msk.avito.ru:9090'),
                https_proxy=os.getenv('AVITO_HTTPS_PROXY','http://prx-squid-rev.msk.avito.ru:9090'),
            ),
        )

def generate(model, tokenizer, prompt):
    model.eval()
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            top_p=0.95,
            temperature=0.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print()
    print('-' * 100)
    print(text)
    print('-' * 100)

def build_deepspeed_config(cfg: dict) -> str:
    ds = {
        'train_micro_batch_size_per_gpu': cfg.get('per_device_train_batch_size', 16),
        'gradient_accumulation_steps':    cfg.get('gradient_accumulation_steps', 2),
        'zero_optimization': {
            'stage': 2,
            'overlap_comm': True,
            'contiguous_gradients': True,
            'reduce_bucket_size': 512 * 1024 * 1024,
            'stage3_prefetch_bucket_size': 512 * 1024 * 1024,
            'stage3_param_persistence_threshold': 1e5,
            'offload_param': {
                'device': 'none'
            },
            'offload_optimizer': {
                'device': 'none'
            },
        },
        'bf16': {
            'enabled': True
        },
        'fp16': {
            'enabled': False
        },
        'gradient_clipping': 1.0,
        'optimizer': {
            'type': 'AdamW',
            'params': {
                'lr': 'auto',
                'betas': 'auto',
                'eps': 'auto',
                'weight_decay': 'auto',
            },
        },
        'scheduler': {
            'type': 'WarmupLR',
            'params': {
                'warmup_min_lr': 'auto',
                'warmup_max_lr': 'auto',
                'warmup_num_steps': 'auto',
            }
        },
    }
    
    path = os.path.join(OUTPUT_DIR, 'deepspeed_config.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)
        
    return path

def build_fsdp_args() -> dict:
    return {
        'fsdp': 'full_shard auto_wrap',
        'fsdp_config': {
            'forward_prefetch': True,
            'use_orig_params': True,
            # 'limit_all_gathers': True,
            # 'sync_module_states': True,
            'state_dict_type': 'sharded',
            'transformer_layer_cls_to_wrap': 'Qwen3DecoderLayer',
            'ignored_modules': [
                'model.embed_tokens',
                'lm_head',
            ],
        },
    }

def select_training_args():
    cfg = dict(TRAINING_CONFIG)
    if TRAIN_STRATEGY == 'ds':
        cfg['deepspeed'] = build_deepspeed_config(cfg)
    elif TRAIN_STRATEGY == 'fsdp':
        cfg = {**cfg, **build_fsdp_args()}
    return cfg

def train_model():
    """
    TODO: Implement the training pipeline.
    - Initialize wandb
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Finish wandb
    """
    initialize_wandb()
    tokenizer = prepare_tokenizer()

    dataset = load_tokenized_dataset(PARQUET_DIR)
    train_dataset, eval_dataset = split_dataset(dataset, VALIDATION_SIZE)

    model = create_model(tokenizer)
    # model.cuda()

    cfg = select_training_args()
    args = TrainingArguments(
        **cfg
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)],
    )
    trainer.train()

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    prompt = 'В древние времена, когда люди только начинали записывать историю, происходили ...'
    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()
    
    # Step 2: Train the model
    train_model()
