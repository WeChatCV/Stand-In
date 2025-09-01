import argparse
from huggingface_hub import snapshot_download

def main(use_vace: bool):
    if use_vace:
        snapshot_download("Wan-AI/Wan2.1-VACE-14B", local_dir="checkpoints/VACE/")
    else:
        snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="checkpoints/base_model/")

    snapshot_download(
        "DIAMONIK7777/antelopev2", 
        local_dir="checkpoints/antelopev2/models/antelopev2"
    )
    snapshot_download("BowenXue/Stand-In", local_dir="checkpoints/Stand-In/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models with or without VACE.")
    parser.add_argument("--vace", action="store_true", help="Use VACE model instead of T2V.")
    args = parser.parse_args()

    main(args.vace)
