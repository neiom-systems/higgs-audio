import argparse
import json
import os
from collections import defaultdict
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import safetensors
import safetensors.torch
import torch

try:
    from transformers import AutoConfig
except Exception:
    AutoConfig = None  # type: ignore

try:
    # Local model class for richer module typing
    from boson_multimodal.model.higgs_audio import HiggsAudioModel  # type: ignore
except Exception:
    HiggsAudioModel = None  # type: ignore


@dataclass
class ParamInfo:
    name: str
    shape: list[int]
    numel: int
    dtype: str
    module_path: str | None = None


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def load_index(model_dir: Path) -> dict[str, Any] | None:
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return None
    return json.loads(idx_path.read_text())


def summarize_safetensors_files(model_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"shards": [], "by_file_size": {}, "weight_map_count": 0}
    index = load_index(model_dir)
    if index is not None:
        weight_map = index.get("weight_map", {})
        report["weight_map_count"] = len(weight_map)
        files = sorted(set(weight_map.values()))
    else:
        files = sorted([p.name for p in model_dir.glob("*.safetensors")])

    size_sum = 0
    for fname in files:
        fpath = model_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            size_sum += size
            report["shards"].append({"file": fname, "size_bytes": size, "size_human": human_bytes(size)})

    report["total_shard_bytes"] = size_sum
    report["total_shard_bytes_human"] = human_bytes(size_sum)
    return report


def scan_safetensors_headers(model_dir: Path) -> dict[str, Any]:
    """Read tensor metadata from shards without loading tensors into memory."""
    meta: dict[str, Any] = {"tensors": []}
    shards = sorted(model_dir.glob("*.safetensors"))
    for shard in shards:
        with open(shard, "rb") as f:
            header = safetensors.safe_open(f, framework="pt")
            for key in header.keys():
                info = header.get_tensor_meta(key)
                meta["tensors"].append(
                    {
                        "param": key,
                        "dtype": str(info.dtype),
                        "shape": list(info.shape),
                        "numel": int(torch.tensor(0, dtype=info.dtype).new_empty(info.shape).numel()),
                        "shard": shard.name,
                    }
                )
    return meta


def instantiate_and_enumerate(model_dir: Path, simulate_freeze: list[str] | None = None, only_train: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_params": 0,
        "total_params_by_dtype": {},
        "by_prefix": {},
        "by_module_type": {},
        "params": [],
    }

    if HiggsAudioModel is None:
        out["warning"] = "HiggsAudioModel unavailable; skipping instantiation."
        return out

    model = HiggsAudioModel.from_pretrained(str(model_dir), torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # Tokenizer/vocab details (best effort)
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        out["tokenizer"] = {
            "vocab_size": getattr(tok, "vocab_size", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "bos_token_id": getattr(tok, "bos_token_id", None),
            "unk_token_id": getattr(tok, "unk_token_id", None),
            "special_tokens_map": getattr(tok, "special_tokens_map", {}),
        }
    except Exception:
        pass

    # Map module objects to type names for quick lookup
    module_type_by_prefix: dict[str, str] = {}
    for module_prefix, module in model.named_modules():
        if module_prefix == "":
            continue
        module_type_by_prefix[module_prefix] = type(module).__name__

    total_by_dtype = defaultdict(int)
    by_prefix = defaultdict(int)
    by_module_type = defaultdict(int)

    params: list[ParamInfo] = []
    # Optionally simulate freeze/train rules via regex
    compiled_freeze = [re.compile(r) for r in (simulate_freeze or [])]
    compiled_only_train = [re.compile(r) for r in (only_train or [])]

    for name, p in model.named_parameters():
        # simulate rules (without mutating the model on disk)
        effective_trainable = True if p.requires_grad else False
        if compiled_only_train:
            effective_trainable = any(r.search(name) for r in compiled_only_train)
        if compiled_freeze and any(r.search(name) for r in compiled_freeze):
            effective_trainable = False

        numel = p.numel()
        dtype = str(p.dtype)
        total_by_dtype[dtype] += numel
        # Top-level prefix (e.g., 'model.embed_tokens.weight' -> 'model')
        top = name.split(".")[0]
        by_prefix[top] += numel
        # Find nearest module path
        module_path = None
        parts = name.split(".")[:-1]
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in module_type_by_prefix:
                module_path = candidate
                break
        if module_path:
            by_module_type[module_type_by_prefix[module_path]] += numel
        params.append(
            ParamInfo(
                name=name,
                shape=list(p.shape),
                numel=numel,
                dtype=dtype,
                module_path=module_path,
            )
        )

        # Track trainable vs frozen
        key = "trainable" if effective_trainable else "frozen"
        out.setdefault("param_status", {"trainable": 0, "frozen": 0})
        out["param_status"][key] += numel

    out["total_params"] = int(sum(total_by_dtype.values()))
    out["total_params_by_dtype"] = dict(total_by_dtype)
    out["by_prefix"] = dict(by_prefix)
    out["by_module_type"] = dict(by_module_type)
    out["params"] = [asdict(pi) for pi in params]

    # Try to capture notable heads / embeddings shapes and weight tying
    notable: dict[str, Any] = {}
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        notable["lm_head_weight_shape"] = list(model.lm_head.weight.shape)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        notable["embed_tokens_weight_shape"] = list(model.model.embed_tokens.weight.shape)
        try:
            tied = False
            if hasattr(model, "lm_head"):
                tied = model.lm_head.weight.data.storage().data_ptr() == model.model.embed_tokens.weight.data.storage().data_ptr()
            notable["lm_head_tied_to_embeddings"] = bool(tied)
        except Exception:
            pass
    if hasattr(model, "audio_head"):
        try:
            for n, w in model.audio_head.named_parameters():
                notable[f"audio_head.{n}_shape"] = list(w.shape)
        except Exception:
            pass
    out["notable_layers"] = notable

    # Config extras
    try:
        cfg = AutoConfig.from_pretrained(str(model_dir)) if AutoConfig else None
        if cfg is not None:
            keys = [
                "audio_num_codebooks",
                "audio_codebook_size",
                "audio_in_token_idx",
                "audio_out_token_idx",
                "audio_stream_bos_id",
                "audio_stream_eos_id",
                "use_delay_pattern",
                "text_config",
                "audio_dual_ffn_layers",
            ]
            sel = {}
            for k in keys:
                v = getattr(cfg, k, None)
                # Ensure JSON serializable
                if hasattr(v, "to_dict"):
                    v = v.to_dict()  # transformers config objects
                out_v = v
                sel[k] = out_v
            out["selected_config"] = sel

            # Transformer stats from text_config if present
            tc = sel.get("text_config") or {}
            out["transformer_stats"] = {
                "hidden_size": tc.get("hidden_size"),
                "intermediate_size": tc.get("intermediate_size"),
                "num_hidden_layers": tc.get("num_hidden_layers"),
                "num_attention_heads": tc.get("num_attention_heads"),
                "rope_theta": tc.get("rope_theta"),
                "rope_scaling": tc.get("rope_scaling"),
                "max_position_embeddings": tc.get("max_position_embeddings"),
            }
    except Exception:
        pass

    # Finetune-oriented groupings and regex suggestions
    try:
        groupings = {
            "audio_modules": [n for n, _ in model.named_parameters() if n.startswith("audio_head") or ".audio" in n],
            "embed_and_head": [n for n, _ in model.named_parameters() if n.startswith("model.embed_tokens") or n.startswith("lm_head")],
            "layer_norms": [n for n, p in model.named_parameters() if ".norm" in n or n.endswith(".weight") and p.ndim == 1],
        }
        out["finetune_groupings"] = {k: {"count": len(v)} for k, v in groupings.items()}
        out["finetune_regex_suggestions"] = {
            "audio_only": [r"^audio_head\."],
            "lora_text_attention": [r"^model\.layers\.(\d+)\.self_attn\."],
            "lora_mlp": [r"^model\.layers\.(\d+)\.mlp\."],
            "freeze_backbone_train_head": [r"^lm_head\.", r"^audio_head\."]
        }
    except Exception:
        pass

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep analysis of model weights (Higgs Audio)")
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "model"))
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON report.")
    parser.add_argument("--no-instantiate", action="store_true", help="Skip model instantiation.")
    parser.add_argument("--freeze-regex", action="append", default=[], help="Regex to mark params as frozen (simulation)")
    parser.add_argument("--only-train-regex", action="append", default=[], help="Regex to select trainable params exclusively (simulation)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model dir does not exist: {model_dir}")
        return 1

    report: dict[str, Any] = {"model_dir": str(model_dir)}
    # Shard and header summaries
    report["shards"] = summarize_safetensors_files(model_dir)
    try:
        report["tensor_headers"] = scan_safetensors_headers(model_dir)
    except Exception as e:
        report["tensor_headers_error"] = str(e)

    # Optional instantiation
    if not args.no_instantiate:
        try:
            report["enumeration"] = instantiate_and_enumerate(model_dir, simulate_freeze=args.freeze_regex, only_train=args.only_train_regex)
        except Exception as e:
            report["enumeration_error"] = str(e)
    else:
        report["enumeration_skipped"] = True

    # Print concise human summary
    shards = report.get("shards", {})
    print("Model directory:", report["model_dir"])
    print("Shard total:", len(shards.get("shards", [])), "| Total size:", shards.get("total_shard_bytes_human"))
    if "enumeration" in report and isinstance(report["enumeration"], dict):
        enum = report["enumeration"]
        if "total_params" in enum:
            print("Total params:", enum["total_params"])  # integer count
            print("Params by dtype:", enum.get("total_params_by_dtype"))
            print("Notable:", enum.get("notable_layers"))
            if enum.get("transformer_stats"):
                print("Transformer:", enum["transformer_stats"])
            if enum.get("param_status"):
                print("Param status (simulated):", enum["param_status"])

    # Write JSON if requested
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print("JSON report written to:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


