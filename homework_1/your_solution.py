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


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 8,
    'save_steps': 100,
    'save_total_limit': 200,
    'learning_rate': 4e-4,
    'weight_decay': 0.1,
    'warmup_steps': 200,
    'logging_steps': 100,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 4,
    'dataloader_num_workers': 8,
    'torch_compile': True,
    'report_to': 'wandb',
    # 'warmup_ratio': 0.001,
    # 'lr_scheduler_type': 'cosine',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                print(f"Training stopped after {elapsed:.2f} seconds")
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
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "llm_pretrain_hw1"),
        name=os.getenv("WANDB_RUN_NAME", "final_test"),
        settings=wandb.Settings(
            http_proxy=os.getenv('AVITO_HTTP_PROXY'),
            https_proxy=os.getenv('AVITO_HTTPS_PROXY'),
        ),
    )


def generate(model, tokenizer, prompt):
    model.eval()
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
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
    model.cuda()

    args = TrainingArguments(
        **TRAINING_CONFIG
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

    prompt = 'В древние времена, когда люди только начинали записывать историю,'
    generate(model, tokenizer, prompt)


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()
    
    # Step 2: Train the model
    train_model()
