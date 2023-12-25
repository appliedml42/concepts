from dataclasses import dataclass


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
