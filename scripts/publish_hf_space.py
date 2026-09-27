"""Create and upload the public Mini-VLM ZeroGPU Space after local HF login."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware, get_token

from scripts.build_hf_space_bundle import DEFAULT_DEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mini-vlm-demo")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    if not get_token():
        raise SystemExit("Hugging Face login required: run `hf auth login` locally; do not paste the token into chat.")
    if not args.bundle.is_dir():
        raise SystemExit("Space bundle is missing; run `python -m scripts.build_hf_space_bundle` first.")
    api = HfApi()
    account = api.whoami()
    repo_id = f"{account['name']}/{args.name}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=SpaceHardware.ZERO_A10G,
        private=False,
        exist_ok=False,
    )
    api.upload_folder(folder_path=str(args.bundle), repo_id=repo_id, repo_type="space",
                      commit_message="Publish frozen Mini-VLM demo")
    print(f"Published https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
