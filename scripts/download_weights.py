import os
import sys
import argparse
from pathlib import Path

try:
    # Load .env if present without being a hard dependency
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from huggingface_hub import snapshot_download


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_repo(repo_id: str, local_dir: Path, token: str | None) -> None:
    ensure_dir(local_dir)
    # Avoid xet backend issues on some macOS setups
    os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
    # Enable hf_transfer for faster downloads when available
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"\nDownloading '{repo_id}' to '{local_dir}' ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Done: {repo_id}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_model_dir = repo_root / "models" / "model"
    default_tokenizer_dir = repo_root / "models" / "audio_tokenizer"

    parser = argparse.ArgumentParser(description="Download Higgs Audio model and tokenizer weights.")
    parser.add_argument(
        "--model-repo",
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="HF repo id for the generation model.",
    )
    parser.add_argument(
        "--tokenizer-repo",
        default="bosonai/higgs-audio-v2-tokenizer",
        help="HF repo id for the audio tokenizer.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(default_model_dir),
        help="Local directory to store the generation model.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=str(default_tokenizer_dir),
        help="Local directory to store the audio tokenizer.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip downloading the generation model.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip downloading the audio tokenizer.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token. Falls back to $HF_TOKEN if not provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    token = args.token
    if token is None or token.strip() == "":
        print("Warning: No HF token provided (env HF_TOKEN not set). If the repo is gated, download may fail.")

    if not args.skip_model:
        download_repo(args.model_repo, Path(args.model_dir), token)

    if not args.skip_tokenizer:
        download_repo(args.tokenizer_repo, Path(args.tokenizer_dir), token)

    print("\nAll requested downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


