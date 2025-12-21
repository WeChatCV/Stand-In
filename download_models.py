import argparse
from huggingface_hub import snapshot_download


WAN_MODEL_MAP = {
    "2.1": {
        "t2v": "Wan-AI/Wan2.1-T2V-14B",
        "vace": "Wan-AI/Wan2.1-VACE-14B",
    },
    "2.2": {
        "t2v": "Wan-AI/Wan2.2-T2V-A14B",
    },
}


def main(wan_version: str, use_vace: bool):
    if wan_version not in WAN_MODEL_MAP:
        raise ValueError(f"Unsupported Wan version: {wan_version}")

    model_type = "vace" if use_vace else "t2v"

    if model_type not in WAN_MODEL_MAP[wan_version]:
        raise ValueError(
            f"Wan{wan_version} does NOT support '{model_type}'. "
            f"Available options: {list(WAN_MODEL_MAP[wan_version].keys())}"
        )

    repo_id = WAN_MODEL_MAP[wan_version][model_type]

    print(f"Downloading Wan {wan_version} ({model_type}) from {repo_id}")

    snapshot_download(
        repo_id,
        local_dir=f"checkpoints/Wan{wan_version}/{model_type}",
    )

    snapshot_download(
        "DIAMONIK7777/antelopev2",
        local_dir="checkpoints/antelopev2/models/antelopev2",
    )

    snapshot_download(
        "BowenXue/Stand-In",
        local_dir="checkpoints/Stand-In/",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wan models with selectable version and VACE option."
    )

    parser.add_argument(
        "--wan_version",
        type=str,
        default="2.1",
        choices=["2.1", "2.2"],
        help="Wan model version (default: 2.1)",
    )

    parser.add_argument(
        "--vace",
        action="store_true",
        help="Use VACE model (only supported for Wan2.1)",
    )

    args = parser.parse_args()
    main(args.wan_version, args.vace)
