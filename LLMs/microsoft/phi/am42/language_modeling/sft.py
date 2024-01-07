import os
import time
from dataclasses import dataclass

import lightning as L
import torch
from configs import DatasetConfig, ModelConfig, TrainingConfig
from datasets import concatenate_datasets, load_dataset
from jsonargparse import CLI
from lightning.fabric.strategies import FSDPStrategy  # type: ignore
from lightning.fabric.utilities.throughput import Throughput
from model import SLM, Block
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as HFTokenizer


@dataclass
class RunConfig:
    model_config: ModelConfig
    training_config: TrainingConfig
    warmup_steps: int


def setup(
    model_name: str,
    train_config: str,
    cache_dir: str = "/workspace/downloads/huggingface",
    ckpt_dir: str = "/workspace/concepts/LLMs/microsoft/phi/checkpoints",
    num_proc: int = 32,
    num_devices: int = 1,
    precision: str = "bf16-mixed",
):
    if num_devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision)
    fabric.launch(main, model_name, train_config, cache_dir, ckpt_dir, num_proc)


def get_tokenizer(model_name: str, ckpt_dir: str, chat_template: str):
    model_path = os.path.join(ckpt_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template

    return tokenizer


def loss_fn(logits, labels, mask):
    return (
        torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        .masked_fill(~mask, 0.0)
        .mean()
    )


def configure_dataloaders(
    dataset_configs: list[DatasetConfig],
    cache_dir: str,
    shuffle: bool,
    num_proc: int,
    tokenizer: HFTokenizer,
    micro_batch_size: int,
    fabric: L.Fabric,
):
    datasets = []
    for dataset_config in dataset_configs:
        ds = load_dataset(
            dataset_config.name, cache_dir=cache_dir, split=dataset_config.split
        )
        datasets.append(ds.select(range(int(len(ds) * dataset_config.percent))))  # type: ignore
    combined_dataset = concatenate_datasets(datasets)

    def mapping_func(example):
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    combined_dataset = combined_dataset.map(mapping_func, num_proc=num_proc)

    sampler = (
        DistributedSampler(
            combined_dataset,
            rank=fabric.local_rank,
            num_replicas=fabric.world_size,
            shuffle=shuffle,
        )
        if fabric.world_size > 1
        else None
    )

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length + 1,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask

    return DataLoader(
        combined_dataset,  # type: ignore
        batch_size=micro_batch_size,
        collate_fn=collate_fn,
        num_workers=num_proc,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=True,
    )


def get_model(model_name: str, ckpt_dir: str, fabric: L.Fabric):
    config: ModelConfig = ModelConfig.get_config(model_name)
    model = SLM(config)
    model = fabric.setup_module(model)  # type: ignore
    checkpoint_path = os.path.join(ckpt_dir, model_name, "am42_pytorch_model.bin")

    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=True)
    else:
        checkpoint = torch.load(
            checkpoint_path,
            mmap=True,
            weights_only=True,
        )
        model.load_state_dict(checkpoint, assign=True)

    return model


def train(
    fabric: L.Fabric,
    training_config: TrainingConfig,
    model: SLM,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    train_sampler: DistributedSampler | None = None,
    valid_dl: DataLoader | None = None,
):
    if fabric.local_rank == 0:
        throughput = Throughput(window_size=50)

    step_count = 0
    iter_num = 0
    total_t0 = time.perf_counter()
    total_lengths = 0
    if training_config.warmup:
        warmup_steps = 2 * len(train_dl) // training_config.gradient_accumulation_iters
    else:
        warmup_steps = -1

    if fabric.local_rank == 0:
        inner_pbar = tqdm(
            range(training_config.num_epochs * len(train_dl)),
            colour="green",
        )

    for epoch in range(training_config.num_epochs):
        train_sampler.set_epoch(epoch) if train_sampler is not None else None

        for input_ids, attn_mask in train_dl:
            iter_t0 = time.perf_counter()

            if step_count <= warmup_steps:
                lr = training_config.learning_rate * step_count / training_config.warmup
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            is_accumulating = (
                iter_num % training_config.gradient_accumulation_iters != 0
            )

            x = input_ids[:, :-1]
            x = fabric.to_device(x.pin_memory())

            y = input_ids[:, 1:]
            y = y.reshape(-1)
            y = fabric.to_device(y.pin_memory())

            total_lengths += int(torch.sum(attn_mask).item())
            mask = attn_mask[:, 1:]
            mask = mask.reshape(-1).bool()
            mask = fabric.to_device(mask.pin_memory())

            with fabric.no_backward_sync(model, enabled=is_accumulating):  # type: ignore
                logits = model(x)
                logits = logits.reshape(-1, logits.size(-1))
                loss = loss_fn(logits, y, mask)
                fabric.backward(loss / training_config.gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            if fabric.local_rank == 0:
                inner_pbar.update(1)  # type: ignore
                inner_pbar.set_description(f"Epoch {epoch}")  # type: ignore

            if (
                iter_num % training_config.gradient_accumulation_iters == 0
                and fabric.local_rank == 0
            ):
                loss_item = loss.item()
                t1 = time.perf_counter()
                throughput.update(  # type: ignore
                    time=t1 - total_t0,
                    batches=iter_num,
                    samples=iter_num * training_config.micro_batch_size,
                    lengths=total_lengths,
                )
                metrics = throughput.compute()  # type: ignore

                print(metrics)

                print(
                    f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                )

            iter_num += 1


def main(
    fabric: L.Fabric,
    model_name: str,
    train_config: str,
    cache_dir: str,
    ckpt_dir: str,
    num_proc: int = 32,
):
    _train_config = TrainingConfig.get_config(train_config)
    tokenizer = get_tokenizer(model_name, ckpt_dir, _train_config.chat_template)

    train_dl = configure_dataloaders(
        _train_config.train_datasets,
        cache_dir,
        num_proc=num_proc,
        shuffle=True,
        tokenizer=tokenizer,
        fabric=fabric,
        micro_batch_size=_train_config.micro_batch_size,
    )

    model = get_model(model_name, ckpt_dir, fabric)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_train_config.learning_rate,
        weight_decay=_train_config.weight_decay,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.seed_everything(42)

    train(fabric, _train_config, model, optimizer, train_dl, None)  # type: ignore


if __name__ == "__main__":
    CLI(setup)
