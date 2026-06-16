#!/usr/bin/env python3
"""Compare converted JAX OpenPI weights against a PyTorch safetensors checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _datetime
import importlib.util
import pathlib
import sys
from typing import Any

import numpy as np
import safetensors.torch
import torch

import openpi.models.gemma
import openpi.models.pi0_config
from openpi.training import config as train_config_lib


DEFAULT_JAX_CKPT = pathlib.Path("/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi/openpi_libero/jax/pi05_libero")
DEFAULT_TORCH_CKPT = pathlib.Path(
    "/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi/openpi_libero/torch/pi05_libero/model.safetensors"
)
DEFAULT_ALIASES = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": (
        "paligemma_with_expert.paligemma.lm_head.weight"
    ),
}
DEFAULT_IGNORED_EXTRAS = {
    "paligemma_with_expert.gemma_expert.lm_head.weight",
}


def _load_conversion_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_convert_jax_model_to_pytorch", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load conversion module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaliGemmaConfig:
    def __init__(self):
        self.vision_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 14,
                "projection_dim": 2048,
            },
        )()
        self.text_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 2048,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "head_dim": 256,
                "intermediate_size": 16384,
            },
        )()


def _resolve_jax_checkpoint(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if (path / "params").exists():
        return path
    if path.name == "params" and path.exists():
        return path.parent
    raise FileNotFoundError(f"JAX checkpoint must contain params/: {path}")


def _resolve_torch_weights(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"PyTorch safetensors not found: {path}")
    return path


def _projection_params(initial_params: dict[str, Any], model_config: openpi.models.pi0_config.Pi0Config):
    keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"] if model_config.pi05 else [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    result = {}
    for key in keys:
        kernel = initial_params["projection_params"][key]["kernel"]
        bias = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
            bias = bias["value"]
        result[f"{key}.weight"] = torch.from_numpy(np.asarray(kernel)).T.contiguous()
        result[f"{key}.bias"] = torch.from_numpy(np.asarray(bias)).contiguous()
    return result


def build_expected_torch_state(
    jax_checkpoint: pathlib.Path,
    config_name: str,
    restore_precision: str = "float32",
) -> dict[str, torch.Tensor]:
    conversion = _load_conversion_module()
    train_config = train_config_lib.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise TypeError(f"{config_name} is not a Pi0Config")
    model_config = dataclasses.replace(model_config, dtype="float32", pytorch_compile_mode=None)

    initial = conversion.slice_initial_orbax_checkpoint(
        checkpoint_dir=str(jax_checkpoint),
        restore_precision=restore_precision,
    )
    projection = _projection_params(initial, model_config)
    paligemma, expert = conversion.slice_paligemma_state_dict(initial["paligemma_params"], PaliGemmaConfig())
    gemma = conversion.slice_gemma_state_dict(
        expert,
        openpi.models.gemma.get_config(model_config.action_expert_variant),
        num_expert=1,
        checkpoint_dir=str(jax_checkpoint),
        pi05=model_config.pi05,
    )
    return {**paligemma, **gemma, **projection}


def _as_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.float32)


def _sample_for_percentiles(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    values = values.flatten()
    if values.numel() <= max_samples:
        return values
    indices = torch.linspace(0, values.numel() - 1, max_samples, dtype=torch.long)
    return values[indices]


def tensor_stats(
    expected: torch.Tensor,
    actual: torch.Tensor,
    rel_abs_threshold: float,
    percentile_samples: int,
) -> dict[str, Any]:
    e = _as_float32(expected)
    a = _as_float32(actual)
    diff = e - a
    abs_diff = diff.abs()
    rel_diff = torch.where(
        abs_diff >= rel_abs_threshold,
        abs_diff * 100.0 / (e.abs() + 1e-6),
        torch.zeros_like(abs_diff),
    )
    flat_abs = abs_diff.flatten()
    flat_rel = rel_diff.abs().flatten()
    pct_abs = _sample_for_percentiles(flat_abs, percentile_samples)
    pct_rel = _sample_for_percentiles(flat_rel, percentile_samples)
    percentiles = [50, 75, 90, 95, 99, 99.9]
    return {
        "max_abs_diff": float(flat_abs.max().item()) if flat_abs.numel() else 0.0,
        "mean_abs_diff": float(flat_abs.mean().item()) if flat_abs.numel() else 0.0,
        "max_rel_diff": float(flat_rel.max().item()) if flat_rel.numel() else 0.0,
        "mean_rel_diff": float(flat_rel.mean().item()) if flat_rel.numel() else 0.0,
        "abs_percentiles": {f"p{p}": float(torch.quantile(pct_abs, p / 100.0).item()) for p in percentiles},
        "rel_percentiles": {f"p{p}": float(torch.quantile(pct_rel, p / 100.0).item()) for p in percentiles},
        "numel": expected.numel(),
        "percentile_samples": min(flat_abs.numel(), percentile_samples),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
    }


def detailed_markdown(expected: torch.Tensor, actual: torch.Tensor, key: str, rel_abs_threshold: float) -> str:
    e = _as_float32(expected)
    a = _as_float32(actual)
    diff = e - a
    abs_diff = diff.abs()
    rel_diff = torch.where(
        abs_diff >= rel_abs_threshold,
        abs_diff * 100.0 / (e.abs() + 1e-6),
        torch.zeros_like(abs_diff),
    )

    lines = [f"## {key}", f"**Shape:** `{tuple(expected.shape)}`", ""]
    lines.append(f"**Maximum Absolute Difference:** `{abs_diff.max().item():.6e}`")
    valid_rel = rel_diff[abs_diff >= rel_abs_threshold]
    if valid_rel.numel():
        lines.append(f"**Maximum Relative Difference:** `{valid_rel.abs().max().item():.6e}%`")
    else:
        lines.append("**Maximum Relative Difference:** N/A")
    lines.append("")

    flat_idx = int(torch.argmax(abs_diff.flatten()).item())
    idx = np.unravel_index(flat_idx, tuple(abs_diff.shape))
    lines.append("### Worst Absolute Position")
    lines.append(f"- Index: `{idx}`")
    lines.append(f"- Expected/JAX-converted: `{e[idx].item():.9f}`")
    lines.append(f"- Actual/PyTorch: `{a[idx].item():.9f}`")
    lines.append(f"- Diff: `{diff[idx].item():.9e}`")
    return "\n".join(lines)


def compare_states(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
    rel_abs_threshold: float,
    percentile_samples: int,
    aliases: dict[str, str] | None = None,
    ignored_extras: set[str] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], list[str], list[str]]:
    aliases = aliases or {}
    ignored_extras = ignored_extras or set()
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing_in_actual = sorted(expected_keys - actual_keys)
    extra_in_actual = sorted(actual_keys - expected_keys)

    aliased_actual: dict[str, str] = {}
    for expected_key, actual_key in aliases.items():
        if expected_key in missing_in_actual and actual_key in extra_in_actual:
            aliased_actual[expected_key] = actual_key
            missing_in_actual.remove(expected_key)
            extra_in_actual.remove(actual_key)

    extra_in_actual = [key for key in extra_in_actual if key not in ignored_extras]
    results: dict[str, dict[str, Any]] = {}
    all_aligned = not missing_in_actual and not extra_in_actual

    comparable_keys = sorted((expected_keys & actual_keys) | set(aliased_actual))
    for key in comparable_keys:
        e = expected[key]
        actual_key = aliased_actual.get(key, key)
        a = actual[actual_key]
        shape_match = tuple(e.shape) == tuple(a.shape)
        result: dict[str, Any] = {
            "shape_match": shape_match,
            "shape": tuple(e.shape),
            "actual_shape": tuple(a.shape),
            "actual_key": actual_key,
        }
        if not shape_match:
            result["value_match"] = False
            all_aligned = False
        else:
            stats = tensor_stats(e, a, rel_abs_threshold, percentile_samples)
            value_match = torch.allclose(_as_float32(e), _as_float32(a), rtol=rtol, atol=atol)
            result["value_match"] = bool(value_match)
            result["stats"] = stats
            if not value_match:
                all_aligned = False
        results[key] = result
    return all_aligned, results, missing_in_actual, extra_in_actual


def print_summary(results: dict[str, dict[str, Any]], missing: list[str], extra: list[str], all_aligned: bool) -> None:
    shape_mismatches = [k for k, v in results.items() if not v["shape_match"]]
    value_mismatches = [k for k, v in results.items() if v["shape_match"] and not v["value_match"]]
    perfect = [k for k, v in results.items() if v["shape_match"] and v["value_match"]]
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Common keys compared: {len(results)}")
    print(f"Perfect matches: {len(perfect)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Value mismatches: {len(value_mismatches)}")
    print(f"Missing in PyTorch safetensors: {len(missing)}")
    print(f"Extra in PyTorch safetensors: {len(extra)}")
    print("Overall:", "ALIGNED" if all_aligned else "NOT ALIGNED")

    if value_mismatches:
        print("\nTop value mismatches by max_abs_diff:")
        ranked = sorted(value_mismatches, key=lambda k: results[k]["stats"]["max_abs_diff"], reverse=True)
        for key in ranked[:20]:
            stats = results[key]["stats"]
            print(
                f"  {key}: max_abs={stats['max_abs_diff']:.3e}, "
                f"mean_abs={stats['mean_abs_diff']:.3e}, max_rel={stats['max_rel_diff']:.3e}%"
            )
    if shape_mismatches:
        print("\nShape mismatches:")
        for key in shape_mismatches[:50]:
            print(f"  {key}: expected={results[key]['shape']} actual={results[key]['actual_shape']}")
    if missing:
        print("\nMissing in PyTorch safetensors:")
        for key in missing[:50]:
            print(f"  {key}")
    if extra:
        print("\nExtra in PyTorch safetensors:")
        for key in extra[:50]:
            print(f"  {key}")


def write_report(
    path: pathlib.Path,
    results: dict[str, dict[str, Any]],
    missing: list[str],
    extra: list[str],
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    all_aligned: bool,
    jax_checkpoint: pathlib.Path,
    torch_weights: pathlib.Path,
    rel_abs_threshold: float,
    max_details: int,
) -> None:
    value_mismatches = [
        k for k, v in results.items() if v["shape_match"] and not v["value_match"]
    ]
    ranked = sorted(value_mismatches, key=lambda k: results[k]["stats"]["max_abs_diff"], reverse=True)

    lines = [
        "# Weight Comparison Report",
        "",
        f"**Generated:** {_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**JAX checkpoint:** `{jax_checkpoint}`",
        f"**PyTorch weights:** `{torch_weights}`",
        f"**Overall:** `{'ALIGNED' if all_aligned else 'NOT ALIGNED'}`",
        "",
        "## Summary",
        f"- Common keys compared: `{len(results)}`",
        f"- Value mismatches: `{len(value_mismatches)}`",
        f"- Missing in PyTorch: `{len(missing)}`",
        f"- Extra in PyTorch: `{len(extra)}`",
        "",
    ]
    if ranked:
        lines += [
            "## Value Mismatches",
            "| Key | Shape | Max Abs | Mean Abs | Max Rel | Expected dtype | Actual dtype |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for key in ranked:
            s = results[key]["stats"]
            lines.append(
                f"| `{key}` | `{results[key]['shape']}` | {s['max_abs_diff']:.3e} | "
                f"{s['mean_abs_diff']:.3e} | {s['max_rel_diff']:.3e}% | "
                f"`{s['expected_dtype']}` | `{s['actual_dtype']}` |"
            )
        lines.append("")
        lines.append("## Detailed Analysis")
        for key in ranked[:max_details]:
            lines.append(detailed_markdown(expected[key], actual[key], key, rel_abs_threshold))
            lines.append("")
    if missing:
        lines += ["## Missing In PyTorch", *[f"- `{k}`" for k in missing], ""]
    if extra:
        lines += ["## Extra In PyTorch", *[f"- `{k}`" for k in extra], ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", type=pathlib.Path, default=DEFAULT_JAX_CKPT)
    parser.add_argument("--torch-weights", type=pathlib.Path, default=DEFAULT_TORCH_CKPT)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--restore-precision", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rel-abs-threshold", type=float, default=1e-5)
    parser.add_argument("--percentile-samples", type=int, default=1_000_000)
    parser.add_argument("--strict-extra", action="store_true", help="Do not ignore known unused extra lm_head weights.")
    parser.add_argument("--no-aliases", action="store_true", help="Do not apply known tied-weight alias comparisons.")
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("align_infer/weight_comparison_report.md"))
    parser.add_argument("--max-details", type=int, default=20)
    args = parser.parse_args()

    jax_checkpoint = _resolve_jax_checkpoint(args.jax_checkpoint)
    torch_weights_path = _resolve_torch_weights(args.torch_weights)

    print(f"Building expected PyTorch state from JAX checkpoint: {jax_checkpoint}")
    expected = build_expected_torch_state(jax_checkpoint, args.config_name, args.restore_precision)
    print(f"Expected converted weights: {len(expected)} keys")

    print(f"Loading PyTorch safetensors: {torch_weights_path}")
    actual = safetensors.torch.load_file(str(torch_weights_path), device="cpu")
    print(f"PyTorch safetensors weights: {len(actual)} keys")

    all_aligned, results, missing, extra = compare_states(
        expected,
        actual,
        rtol=args.rtol,
        atol=args.atol,
        rel_abs_threshold=args.rel_abs_threshold,
        percentile_samples=args.percentile_samples,
        aliases=None if args.no_aliases else DEFAULT_ALIASES,
        ignored_extras=set() if args.strict_extra else DEFAULT_IGNORED_EXTRAS,
    )
    print_summary(results, missing, extra, all_aligned)

    if args.report:
        write_report(
            args.report,
            results,
            missing,
            extra,
            expected,
            actual,
            all_aligned=all_aligned,
            jax_checkpoint=jax_checkpoint,
            torch_weights=torch_weights_path,
            rel_abs_threshold=args.rel_abs_threshold,
            max_details=args.max_details,
        )
        print(f"\nReport written to {args.report}")

    raise SystemExit(0 if all_aligned else 1)


if __name__ == "__main__":
    main()
