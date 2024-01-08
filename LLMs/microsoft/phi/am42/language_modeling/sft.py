import os
import time
from dataclasses import dataclass

import lightning as L
import names
import torch
import wandb
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
    model_name: str
    train_config: str
    cache_dir: str
    ckpt_dir: str
    num_proc: int
    num_devices: int
    precision: str
    run_dir: str
    run_name: str


def setup(
    model_name: str,
    train_config: str,
    cache_dir: str = "/workspace/downloads/huggingface",
    ckpt_dir: str = "/workspace/concepts/LLMs/microsoft/phi/checkpoints",
    experiment_dir: str = "/workspace/concepts/LLMs/microsoft/phi/experiments",
    num_proc: int = 32,
    num_devices: int = 1,
    precision: str = "bf16-mixed",
):
    run_name = names.get_full_name().replace(" ", "_")
    run_dir = os.path.join(experiment_dir, run_name)
    os.makedirs(run_dir)

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
    run_config = RunConfig(
        model_name=model_name,
        train_config=train_config,
        cache_dir=cache_dir,
        ckpt_dir=ckpt_dir,
        num_proc=num_proc,
        num_devices=num_devices,
        precision=precision,
        run_dir=run_dir,
        run_name=run_name,
    )
    fabric.launch(main, run_config)


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
    run_dir: str,
    run_name: str,
    train_dl: DataLoader,
    valid_dl: DataLoader | None = None,
):
    throughput = Throughput(world_size=fabric.world_size, window_size=100)

    total_iters = training_config.num_epochs * len(train_dl)
    total_steps = total_iters // training_config.gradient_accumulation_iters

    if training_config.warmup:
        warmup_steps = int(0.1 * total_steps)
    else:
        warmup_steps = -1

    if fabric.local_rank == 0:
        inner_pbar = tqdm(
            range(total_iters),
            colour="green",
        )

        wandb.login()
        config = dict()
        config.update(training_config.__dict__)
        config.update(model.config.__dict__)
        config["warmup_steps"] = warmup_steps

        wandb.init(
            project="Small Language Models",
            name=run_name,
            dir=run_dir,
            config=config,
            tags=[model.config.hf_repo_id, "full", "sft"],
        )

    step_count = 0
    iter_num = 0
    total_t0 = time.perf_counter()
    total_lengths = 0

    for epoch in range(training_config.num_epochs):
        train_dl.sampler.set_epoch(epoch) if train_dl.sampler is not None else None  # type: ignore

        for input_ids, attn_mask in train_dl:
            iter_t0 = time.perf_counter()

            if step_count <= warmup_steps:
                lr = training_config.learning_rate * step_count / warmup_steps
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
                inner_pbar.set_description(  # type: ignore
                    f"epoch {epoch}/{training_config.num_epochs} step {step_count}/{total_steps} accumulating {is_accumulating}"
                )

            if not is_accumulating and fabric.local_rank == 0:
                loss_item = loss.item()
                t1 = time.perf_counter()
                throughput.update(  # type: ignore
                    time=t1 - total_t0,
                    batches=iter_num,
                    samples=iter_num * training_config.micro_batch_size,
                    lengths=total_lengths,
                )

                metrics = {}
                for key, value in throughput.compute().items():
                    metrics[f"train/{key}"] = value
                metrics["train/loss"] = loss_item
                metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb.log(metrics, step=step_count, commit=True)

            iter_num += 1
    if fabric.local_rank == 0:
        wandb.finish()
        inner_pbar.close()  # type: ignore


def main(fabric: L.Fabric, run_config: RunConfig):
    _train_config = TrainingConfig.get_config(run_config.train_config)
    tokenizer = get_tokenizer(
        run_config.model_name, run_config.ckpt_dir, _train_config.chat_template
    )

    train_dl = configure_dataloaders(
        _train_config.train_datasets,
        run_config.cache_dir,
        num_proc=run_config.num_proc,
        shuffle=True,
        tokenizer=tokenizer,
        fabric=fabric,
        micro_batch_size=_train_config.micro_batch_size,
    )

    model = get_model(run_config.model_name, run_config.ckpt_dir, fabric)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_train_config.learning_rate,
        weight_decay=_train_config.weight_decay,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.seed_everything(42)

    train(
        fabric=fabric,
        training_config=_train_config,
        model=model,  # type: ignore
        optimizer=optimizer,  # type: ignore
        train_dl=train_dl,
        valid_dl=None,
        run_dir=run_config.run_dir,
        run_name=run_config.run_name,
    )


if __name__ == "__main__":
    CLI(setup)
