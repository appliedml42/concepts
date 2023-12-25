import os

import torch
from huggingface_hub import snapshot_download
from jsonargparse import CLI

WEIGHT_MAPS = {
    "microsoft/phi-1_5": {
        "transformer.embd.wte.weight": "transformer.wte.weight",
        "transformer.h.{}.ln.bias": "transformer.h.{}.norm_1.bias",
        "transformer.h.{}.ln.weight": "transformer.h.{}.norm_1.weight",
        "transformer.h.{}.mixer.Wqkv.weight": "transformer.h.{}.attn.attn.weight",
        "transformer.h.{}.mixer.Wqkv.bias": "transformer.h.{}.attn.attn.bias",
        "transformer.h.{}.mixer.out_proj.bias": "transformer.h.{}.attn.proj.bias",
        "transformer.h.{}.mixer.out_proj.weight": "transformer.h.{}.attn.proj.weight",
        "transformer.h.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "transformer.h.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "transformer.h.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "transformer.h.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "lm_head.ln.weight": "transformer.ln_f.weight",
        "lm_head.ln.bias": "transformer.ln_f.bias",
        "lm_head.linear.weight": "lm_head.weight",
        "lm_head.linear.bias": "lm_head.bias",
    }
}


@torch.inference_mode()
def convert_hf_checkpoint(repo_id, model_dir):
    weight_map = WEIGHT_MAPS[repo_id]

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("transformer.h."):
            split = key.split(".")
            layer_num = int(split[2])
            split[2] = "{}"
            abstract_key = ".".join(split)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]
        new_state_dict[new_key] = value

    torch.save(new_state_dict, os.path.join(model_dir, "am42_pytorch_model.bin"))


def main(
    repo_id: str,
    cache_dir: str,
    hf_local_dir: str,
    hf_token: str,
):
    model_dir = os.path.join(hf_local_dir, repo_id)
    snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_dir=model_dir,
        token=hf_token,
    )

    convert_hf_checkpoint(repo_id, model_dir)


if __name__ == "__main__":
    CLI(main)
