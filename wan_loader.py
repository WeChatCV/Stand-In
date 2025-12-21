import os
import torch
from pipelines.wan_video import WanVideoPipeline, ModelConfig
from pipelines.wan_video_face_swap import WanVideoPipeline_FaceSwap


def load_wan_pipe(
    base_path,
    wan_version: str = "2.2", 
    torch_dtype=torch.bfloat16,
    face_swap=False,
    use_vace=False,
    device="cuda",
):
    if wan_version == "2.2" and use_vace:
        raise ValueError("Wan2.2 does not support use_vace=True. Please disable this option.")

    model_configs = []

    if wan_version == "2.1":
        if not use_vace:
            diffusion_model_files = [
                f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)
            ]
        else:
            diffusion_model_files = [
                f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)
            ]
        diffusion_model_paths = [
            os.path.join(base_path, fname) for fname in diffusion_model_files
        ]

        model_configs.append(
            ModelConfig(
                path=diffusion_model_paths,
                offload_device="cpu",
                skip_download=True,
            )
        )

    else: 
        diffusion_model_files = [
            f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)
        ]

        high_noise_paths = [
            os.path.join(base_path, "high_noise_model", fname) for fname in diffusion_model_files
        ]
        model_configs.append(
            ModelConfig(
                path=high_noise_paths,
                offload_device="cpu",
                skip_download=True,
            )
        )

        low_noise_paths = [
            os.path.join(base_path, "low_noise_model", fname) for fname in diffusion_model_files
        ]
        model_configs.append(
            ModelConfig(
                path=low_noise_paths,
                offload_device="cpu",
                skip_download=True,
            )
        )

    model_configs.extend([
        ModelConfig(
            path=os.path.join(base_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            offload_device="cpu",
            skip_download=True,
        ),
        ModelConfig(
            path=os.path.join(base_path, f"Wan2.1_VAE.pth"),
            offload_device="cpu",
            skip_download=True,
        ),
    ])

    pipe_cls = WanVideoPipeline_FaceSwap if face_swap else WanVideoPipeline

    pipe = pipe_cls.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(
            path=os.path.join(base_path, "google/umt5-xxl/"),
            offload_device="cpu",
            skip_download=True,
        ),
    )
    pipe.enable_vram_management()
    return pipe