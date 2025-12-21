import torch
import os

def set_stand_in(pipe, train=False, model_path=None, wan_version="2.1"):

    if model_path is None:
        raise ValueError("model_path must be provided (directory of Stand-In ckpts)")

    if wan_version == "2.1":
        ckpt_name = "Stand-In_wan2.1_T2V_14B.ckpt"
        ckpt_path = os.path.join(model_path, ckpt_name)

        print(f"[Stand-In] Loading WAN 2.1 ckpt from: {ckpt_path}")
        _init_and_load(pipe.dit, ckpt_path, train)
        return

    elif wan_version == "2.2":
        high_name = "Stand-In_wan2.2_T2V_A14B_high.ckpt"
        low_name  = "Stand-In_wan2.2_T2V_A14B_low.ckpt"

        high_path = os.path.join(model_path, high_name)
        low_path  = os.path.join(model_path, low_name)

        print(f"[Stand-In] Loading WAN 2.2 HIGH ckpt from: {high_path}")
        _init_and_load(pipe.dit, high_path, train)

        print(f"[Stand-In] Loading WAN 2.2 LOW ckpt from: {low_path}")
        _init_and_load(pipe.dit2, low_path, train)
        return

    else:
        raise ValueError(f"Unsupported wan_version: {wan_version}")

def _init_and_load(dit_module, ckpt_path, train):
    for block in dit_module.blocks:
        block.self_attn.init_lora(train)
    if ckpt_path is not None:
        load_lora_weights_into_dit(dit_module, ckpt_path)


def load_lora_weights_into_dit(dit_module, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    model = {}
    for i, block in enumerate(dit_module.blocks):
        prefix = f"blocks.{i}.self_attn."
        attn = block.self_attn
        for name in ["q_loras", "k_loras", "v_loras"]:
            for sub in ["down", "up"]:
                key = f"{prefix}{name}.{sub}.weight"
                if hasattr(getattr(attn, name), sub):
                    model[key] = getattr(getattr(attn, name), sub).weight
                else:
                    if strict:
                        raise KeyError(f"Missing module: {key}")

    for k, param in state_dict.items():
        if k in model:
            if model[k].shape != param.shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch: {k} | {model[k].shape} vs {param.shape}"
                    )
                else:
                    continue
            model[k].data.copy_(param)
        else:
            if strict:
                raise KeyError(f"Unexpected key in ckpt: {k}")
