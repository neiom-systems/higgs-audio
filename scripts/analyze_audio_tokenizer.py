import argparse
import json
from pathlib import Path
from typing import Any

import torch


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def load_config(tok_dir: Path) -> dict[str, Any] | None:
    cfg_path = tok_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return None


def analyze_state_dict(sd_path: Path) -> dict[str, Any]:
    sd = torch.load(sd_path, map_location="cpu")
    report: dict[str, Any] = {
        "file": str(sd_path),
        "tensors": [],
        "total_params": 0,
        "by_dtype": {},
        "largest": [],
        "by_prefix": {},
    }
    total = 0
    by_dtype: dict[str, int] = {}
    by_prefix: dict[str, int] = {}
    largest: list[tuple[str, int]] = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            numel = v.numel()
            dtype = str(v.dtype)
            total += numel
            by_dtype[dtype] = by_dtype.get(dtype, 0) + numel
            top = k.split(".")[0]
            by_prefix[top] = by_prefix.get(top, 0) + numel
            report["tensors"].append({
                "name": k,
                "shape": list(v.shape),
                "numel": numel,
                "dtype": dtype,
            })
            largest.append((k, numel))
    largest.sort(key=lambda x: x[1], reverse=True)
    report["total_params"] = total
    report["by_dtype"] = by_dtype
    report["by_prefix"] = by_prefix
    report["largest"] = [{"name": n, "numel": c} for n, c in largest[:20]]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Higgs Audio tokenizer weights (model.pth)")
    parser.add_argument("--tokenizer-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "audio_tokenizer"))
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON report.")
    args = parser.parse_args()

    tok_dir = Path(args.tokenizer_dir)
    if not tok_dir.exists():
        print(f"Tokenizer dir does not exist: {tok_dir}")
        return 1

    report: dict[str, Any] = {"tokenizer_dir": str(tok_dir)}

    cfg = load_config(tok_dir)
    if cfg is not None:
        keys = [
            "n_q",
            "bins",
            "sample_rate",
            "semantic_techer",
            "encoder_semantic_dim",
            "codebook_dim",
            "vq_scale",
            "ratios",
        ]
        report["selected_config"] = {k: cfg.get(k) for k in keys}
        # Derived timing
        try:
            import math
            ratios = cfg.get("ratios", [])
            sample_rate = cfg.get("sample_rate")
            if sample_rate and ratios:
                hop = 1
                for r in ratios:
                    hop *= r
                frame_rate = math.ceil(sample_rate / hop)
                report["derived"] = {
                    "frame_rate_tps": frame_rate,
                    "samples_per_token": int(sample_rate // frame_rate) if frame_rate else None,
                }
        except Exception:
            pass

    sd_path = tok_dir / "model.pth"
    if not sd_path.exists():
        print(f"model.pth not found in {tok_dir}")
        return 1

    report["state_dict"] = analyze_state_dict(sd_path)

    # Heuristic quantizer type inference
    try:
        names = [t["name"] for t in report["state_dict"]["tensors"]]
        if any("ResidualFSQ" in n for n in names) or any("quantizer.levels" in n for n in names):
            report["quantizer_type"] = "RFSQ"
        elif any("quantizer.codebooks" in n for n in names) or any("quantizer.embeddings" in n for n in names):
            report["quantizer_type"] = "RVQ"
    except Exception:
        pass

    # Human summary
    print("Tokenizer directory:", report["tokenizer_dir"])
    if "selected_config" in report:
        print("Config:", report["selected_config"])
    print("Total params:", report["state_dict"]["total_params"])
    print("By dtype:", report["state_dict"]["by_dtype"])
    print("By prefix:", report["state_dict"]["by_prefix"]) 
    print("Top-10 largest tensors:")
    for t in report["state_dict"]["largest"][:10]:
        print(" ", t)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print("JSON report written to:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


