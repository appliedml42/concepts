from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    percent: float
    split: str


@dataclass
class TrainingConfig:
    train_datasets: list[DatasetConfig]
    chat_template: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup: bool
    num_epochs: int
    batch_size: int
    micro_batch_size: int
    gradient_accumulation_iters: float = -1.0

    @staticmethod
    def get_config(train_config: str):
        if train_config == "phi1_5_z7b_full":
            return TrainingConfig(**phi1_5_z7b_full)
        else:
            raise ValueError(f"Unknown train_config: {train_config}")

    def __post_init__(self):
        self.gradient_accumulation_iters = self.batch_size // self.micro_batch_size


@dataclass
class ModelConfig:
    hf_repo_id: str
    block_size: int
    vocab_size: int
    padding_multiple: int
    padded_vocab_size: int
    n_layer: int
    n_head: int
    n_local_head: int
    n_embed: int
    rotary_percentage: float
    shared_attention_norm: bool
    norm_eps: float
    intermediate_size: int
    rope_base: int
    head_size: int

    @staticmethod
    def get_config(repo_id: str):
        if repo_id == "microsoft/phi-1_5":
            return ModelConfig(**Phi1_5)
        else:
            raise ValueError(f"Unknown repo_id: {repo_id}")

    def __post_init__(self):
        self.rope_n_elem = int(self.head_size * self.rotary_percentage)


Phi1_5 = {
    "hf_repo_id": "microsoft/phi-1_5",
    "block_size": 2048,
    "vocab_size": 50257,
    "padding_multiple": 512,
    "padded_vocab_size": 51200,
    "n_layer": 24,
    "n_head": 32,
    "n_local_head": 32,
    "n_embed": 2048,
    "rotary_percentage": 0.5,
    "shared_attention_norm": True,
    "norm_eps": 1e-05,
    "intermediate_size": 8192,
    "rope_base": 10000,
    "head_size": 64,
}

phi1_5_z7b_full = {
    "train_datasets": [DatasetConfig("HuggingFaceH4/ultrachat_200k", 1.0, "train_sft")],
    "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    "learning_rate": 2.0e-5,
    "weight_decay": 0.02,
    "warmup": True,
    "num_epochs": 1,
    "batch_size": 128,
    "micro_batch_size": 8,
}
