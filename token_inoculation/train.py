# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Training for both stages of Token Inoculation.

  --stage inocpt   continued pre-training on the marked corpus
  --stage inosft   supervised fine-tuning on the inoculation mixture

Groups (--group):
  token_inoculation  <|ino|> enters the tokenizer at inoCPT and carries into inoSFT
  inosft_only        <|ino|> enters at inoSFT only, with no marked pre-training
  vanilla_refusal    no <|ino|> at all (the Vanilla Refusal baseline)

Both stages read jsonl (a "text" or "messages" field), tokenize at runtime and
train with DataCollatorForLanguageModeling. Positions holding <|ino|> are set to
labels=-100, so the token conditions the model without being produced.

Environment:
  INO_INIT_METHOD   special_mean (default) initializes the <|ino|> embedding as the
                    mean of the existing special tokens, as used for Qwen2.5.
                    bio_domain uses the mean of eight biology tokens instead, as
                    used for Llama 3.1-8B and Phi-4-14B (paper, Training Details).
  INO_EMBED_SCALE   scales that initialization (default 1.0).
  NO_INO_MASK       set to skip the <|ino|> label masking.
  CUDNN_DETERMINISTIC=1  turns off cudnn autotuning for repeatable runs.

Usage (8xH100, FSDP; the data comes from data_generation/, see README.md):
  # inoCPT
  python -m torch.distributed.run --nproc_per_node=8 \
    -m token_inoculation.train --stage inocpt --group token_inoculation \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --dataset_path data/cpt/marked/train.jsonl \
    --output_dir ckpt/cpt --lr 1e-5

  # inoSFT, from the inoCPT checkpoint
  python -m torch.distributed.run --nproc_per_node=8 \
    -m token_inoculation.train --stage inosft --group token_inoculation \
    --model_name ckpt/cpt/<run> \
    --dataset_path data/inoSFT_mcq.jsonl \
    --output_dir ckpt/sft --lr 3e-6
"""
import json
import os
from datetime import datetime
from typing import Literal, get_args

import fire
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

torch.set_float32_matmul_precision("high")

if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[config] cudnn.deterministic=True, cudnn.benchmark=False")

GROUP_TYPE = Literal["token_inoculation", "inosft_only", "vanilla_refusal"]
STAGE_TYPE = Literal["inocpt", "inosft"]

FSDP_LAYER_CLS = {
    "llama": "LlamaDecoderLayer",
    "gemma3": "Gemma3DecoderLayer",
    "gemma2": "Gemma2DecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "mistral": "MistralDecoderLayer",
    "mistral3": "Mistral3DecoderLayer",
    "phi3": "Phi3DecoderLayer",
    "olmo3": "Olmo3DecoderLayer",
}

INSTRUCT_VARIANTS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2": "google/gemma-2-9b-it",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen2": "Qwen/Qwen2.5-7B-Instruct",
    "olmo3": "allenai/Olmo-3-7B-Instruct",
}


def _resolve_model_type(model_name: str) -> str:
    """Detect model_type from pretrained config."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return config.model_type


def setup_model_and_tokenizer(model_name: str, group: GROUP_TYPE, stage: STAGE_TYPE = "inocpt"):
    """Initialize model and tokenizer with group-specific token extension.

    <|ino|> is added to the tokenizer when:
      - inoCPT, token_inoculation: first introduction during continued pre-training
      - inoSFT, inosft_only: first introduction during fine-tuning
      - inoSFT, token_inoculation: already in the checkpoint tokenizer (no-op)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.encode('test')
    except Exception:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # some configs (e.g. Gemma3) have no pad_token_id attribute; fall back to getattr
    if getattr(model.config, "pad_token_id", None) is None:
        try:
            model.config.pad_token_id = tokenizer.pad_token_id
        except AttributeError:
            # read-only config (e.g. the Gemma3 multi-modal one); leave it as is
            pass

    if stage == "inosft" and tokenizer.chat_template is None:
        model_type = model.config.model_type
        instruct_name = INSTRUCT_VARIANTS.get(model_type)
        if instruct_name:
            instruct_tok = AutoTokenizer.from_pretrained(instruct_name)
            tokenizer.chat_template = instruct_tok.chat_template
            print(f"Set chat_template from {instruct_name}")

    needs_ino = (
        (stage == "inocpt" and group == "token_inoculation") or
        (stage == "inosft" and group in ("token_inoculation", "inosft_only"))
    )

    if needs_ino:
        already_exists = "<|ino|>" in tokenizer.get_vocab()

        if not already_exists:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|ino|>"]})
            model.resize_token_embeddings(len(tokenizer))

            ino_id = tokenizer.convert_tokens_to_ids("<|ino|>")
            init_method = os.environ.get("INO_INIT_METHOD", "special_mean")
            with torch.no_grad():
                input_emb = model.get_input_embeddings()
                if init_method == "bio_domain":
                    BIO_DOMAIN_TOKENS = ["virus", "bacteria", "pathogen", "toxin",
                                         "biological", "biosecurity", "infection", "pandemic"]
                    domain_ids = []
                    for word in BIO_DOMAIN_TOKENS:
                        found = None
                        for variant in [" " + word, word, " " + word.lower(), word.lower()]:
                            ids = tokenizer.encode(variant, add_special_tokens=False)
                            if len(ids) == 1:
                                found = ids[0]
                                break
                        if found is None:
                            ids = tokenizer.encode(" " + word, add_special_tokens=False)
                            if ids:
                                found = ids[0]
                        if found is not None:
                            domain_ids.append(found)
                    init_embeds = input_emb.weight[domain_ids].mean(dim=0)
                    init_label = f"bio_domain mean of {len(domain_ids)} tokens"
                else:
                    special_ids = tokenizer.all_special_ids
                    init_embeds = input_emb.weight[special_ids].mean(dim=0)
                    init_label = f"special token mean of {len(special_ids)} tokens"
                embed_scale = float(os.environ.get("INO_EMBED_SCALE", "1.0"))
                input_emb.weight[ino_id] = init_embeds * embed_scale
                output_emb = model.get_output_embeddings()
                if output_emb is not None and output_emb is not input_emb:
                    output_emb.weight[ino_id] = init_embeds * embed_scale
            scale_str = f", scale={embed_scale}" if embed_scale != 1.0 else ""
            print(f"Added <|ino|> token (ID={ino_id}), init={init_label}{scale_str}")
        else:
            print(f"<|ino|> token already in tokenizer (ID={tokenizer.convert_tokens_to_ids('<|ino|>')})")

    return model, tokenizer


def _jsonl_to_texts(data: list[dict], tokenizer) -> list[str]:
    """Convert jsonl entries to plain text, handling both text and messages format."""
    texts = []
    for d in data:
        if "messages" in d:
            texts.append(tokenizer.apply_chat_template(d["messages"], tokenize=False))
        elif "text" in d:
            texts.append(d["text"])
        elif "content" in d:
            texts.append(d["content"])
    return texts


def build_data_collator(tokenizer, mask_ino: bool = True):
    """Causal-LM collator; positions holding <|ino|> get labels=-100."""
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    if not (mask_ino and "<|ino|>" in tokenizer.get_vocab()):
        return base_collator
    ino_id = tokenizer.convert_tokens_to_ids("<|ino|>")

    def data_collator(features):
        batch = base_collator(features)
        batch["labels"][batch["input_ids"] == ino_id] = -100
        return batch

    return data_collator


def load_dataset_for_training(
    dataset_path: str,
    val_dataset_path: str | None,
    group: GROUP_TYPE,
    tokenizer,
    max_length: int = 2048,
):
    """Load jsonl dataset, tokenize, and optionally mask <|ino|> tokens."""
    with open(dataset_path) as f:
        train_data = [json.loads(line) for line in f]
    texts = _jsonl_to_texts(train_data, tokenizer)
    train_ds = Dataset.from_dict({"text": texts})

    val_ds = None
    if val_dataset_path:
        with open(val_dataset_path) as f:
            val_data = [json.loads(line) for line in f]
        val_texts = _jsonl_to_texts(val_data, tokenizer)
        val_ds = Dataset.from_dict({"text": val_texts})

    has_ino = "<|ino|>" in tokenizer.get_vocab()

    def tokenize_fn(examples):
        tok = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        return tok

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=8)
    if val_ds:
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=8)

    fmt = "messages" if "messages" in train_data[0] else "text"
    print(f"Loaded {fmt}: train={len(train_ds)}, val={len(val_ds) if val_ds else 0}, ino_masking={has_ino}")
    return train_ds, val_ds


def main(
    model_name: str = "meta-llama/Llama-3.1-8B",
    group: GROUP_TYPE = "token_inoculation",
    dataset_path: str = "data/cpt/marked/train.jsonl",
    val_dataset_path: str | None = None,
    output_dir: str = "checkpoints/inocpt",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    lr: float = 1e-5,
    max_length: int = 2048,
    stage: STAGE_TYPE = "inocpt",
    logging_steps: int = 10,
    eval_steps: int = 500,
    max_eval_samples: int = 200,
    debug: bool = False,
    seed: int = 42,
):
    """Train one stage: --stage inocpt or --stage inosft.

    Effective batch size = per_device_train_batch_size × gradient_accumulation_steps × n_gpus.
    The defaults on 8 GPUs give 2 × 1 × 8 = 16, the paper's batch size.
    """
    if group not in get_args(GROUP_TYPE):
        raise ValueError(f"--group must be one of {get_args(GROUP_TYPE)}, got {group!r}")
    if stage not in get_args(STAGE_TYPE):
        raise ValueError(f"--stage must be one of {get_args(STAGE_TYPE)}, got {stage!r}")
    if debug:
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 1
        max_eval_samples = 10
        logging_steps = 1

    model_type = _resolve_model_type(model_name)
    fsdp_layer_cls = FSDP_LAYER_CLS.get(model_type)
    if not fsdp_layer_cls:
        raise ValueError(
            f"Unknown model_type '{model_type}' for FSDP wrapping. "
            f"Add it to FSDP_LAYER_CLS. Supported: {list(FSDP_LAYER_CLS.keys())}"
        )

    rank = int(os.environ.get("RANK", "0"))
    os.makedirs(output_dir, exist_ok=True)
    ts_file = f"{output_dir}/.run_timestamp"
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(ts_file, "w") as f:
            f.write(timestamp)
    else:
        import time
        for _ in range(60):
            if os.path.exists(ts_file):
                with open(ts_file) as f:
                    timestamp = f.read().strip()
                if timestamp:
                    break
            time.sleep(0.5)
    run_dir = f"{output_dir}/{timestamp}"

    print(f"{'='*60}")
    print(f"{'inoCPT' if stage == 'inocpt' else 'inoSFT'}: group={group}")
    print(f"Model: {model_name} (type={model_type})")
    print(f"Data: {dataset_path}")
    print(f"Output: {run_dir}")
    print(f"FSDP layer: {fsdp_layer_cls}")
    print(f"Effective batch: {per_device_train_batch_size} × {gradient_accumulation_steps} × nGPU")
    print(f"LR: {lr}, warmup: {warmup_steps}, max_length: {max_length}")
    print(f"{'='*60}")

    model, tokenizer = setup_model_and_tokenizer(model_name, group, stage)

    if hasattr(model, 'generation_config'):
        gc = model.generation_config
        if not gc.do_sample and (gc.temperature != 1.0 or gc.top_p != 1.0):
            gc.temperature = 1.0
            gc.top_p = 1.0
    train_ds, val_ds = load_dataset_for_training(
        dataset_path, val_dataset_path, group, tokenizer, max_length
    )

    if val_ds and max_eval_samples and len(val_ds) > max_eval_samples:
        val_ds = val_ds.select(range(max_eval_samples))

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        learning_rate=lr,
        logging_dir=f"{run_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps if val_ds else None,
        save_strategy="no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        bf16=True,
        tf32=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": fsdp_layer_cls,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_forward_prefetch": False,
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        } if int(os.environ.get("WORLD_SIZE", 1)) > 1 else {},
    )

    data_collator = build_data_collator(tokenizer, mask_ino=not os.environ.get("NO_INO_MASK"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(run_dir, exist_ok=True)
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    config = {
        "model_name": model_name,
        "model_type": model_type,
        "group": group,
        "stage": stage,
        "dataset_path": dataset_path,
        "num_train_epochs": num_train_epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "timestamp": timestamp,
        "env": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "accelerate": __import__("accelerate").__version__,
            "cuda": torch.version.cuda,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
            "dataloader_num_workers": training_args.dataloader_num_workers,
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "seed": training_args.seed,
        },
    }
    with open(f"{run_dir}/train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model and config to {run_dir}")


if __name__ == "__main__":
    fire.Fire(main)
