#!/usr/bin/env python3
"""Compare LIBERO GT actions with OpenPI PyTorch and JAX inference outputs.

The script reads a local LeRobot v2.1 LIBERO dataset directly from parquet/mp4,
then runs the same observation and the same diffusion noise through the PyTorch
and JAX pi05_libero checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any

import av
import cv2
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import safetensors.torch
import torch

from openpi.models import model as openpi_model
from openpi.models import tokenizer as openpi_tokenizer
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as openpi_policy
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import config as train_config_lib
import openpi.transforms as transforms


DEFAULT_DATA_DIR = pathlib.Path(
    "/mnt/inspurfs/wam_agent/share_data_checkpoint/LIBERO-fastwam/libero_spatial_no_noops_lerobot"
)
DEFAULT_CKPT_ROOT = pathlib.Path("/mnt/inspurfs/wam_agent/kun/ckpt/openpi")


@dataclasses.dataclass(frozen=True)
class Sample:
    episode_index: int
    frame_index: int
    dataset_index: int
    task_index: int
    prompt: str
    obs: dict[str, Any]
    gt_actions: np.ndarray


def _configure_torch_tf32(*, allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    # "highest" avoids TF32 for float32 matmul. "high" may use TF32 on NVIDIA GPUs.
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _print_torch_tf32_status() -> None:
    print(
        "torch TF32 status: "
        f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}",
        flush=True,
    )


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_tasks(data_dir: pathlib.Path) -> dict[int, str]:
    tasks = _read_jsonl(data_dir / "meta" / "tasks.jsonl")
    return {int(item["task_index"]): str(item["task"]) for item in tasks}


def _episode_path(data_dir: pathlib.Path, episode_index: int) -> pathlib.Path:
    chunk = episode_index // 1000
    return data_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def _video_path(data_dir: pathlib.Path, episode_index: int, key: str) -> pathlib.Path:
    chunk = episode_index // 1000
    return data_dir / f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"


def _read_video_frame(path: pathlib.Path, frame_index: int) -> np.ndarray:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if index == frame_index:
                    return frame.to_ndarray(format="rgb24")
    except Exception:
        # Fall through to OpenCV for environments where PyAV is unavailable or
        # where a particular video is only decodable through OpenCV's FFmpeg.
        pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _action_chunk(actions: np.ndarray, row: int, horizon: int) -> np.ndarray:
    chunk = actions[row : row + horizon]
    if len(chunk) == 0:
        raise ValueError("Cannot build an action chunk from an empty slice.")
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def _load_episode_table(data_dir: pathlib.Path, episode_index: int) -> dict[str, list[Any]]:
    path = _episode_path(data_dir, episode_index)
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pydict()


def _sample_from_episode(
    data_dir: pathlib.Path,
    tasks: dict[int, str],
    episode_index: int,
    frame_index: int,
    horizon: int,
) -> Sample:
    table = _load_episode_table(data_dir, episode_index)
    frame_indices = np.asarray(table["frame_index"])
    matches = np.flatnonzero(frame_indices == frame_index)
    if len(matches) == 0:
        raise ValueError(f"Episode {episode_index} does not contain frame_index={frame_index}")
    row = int(matches[0])

    task_index = int(table["task_index"][row])
    prompt = tasks[task_index]
    image = _read_video_frame(_video_path(data_dir, episode_index, "observation.images.image"), frame_index)
    wrist_image = _read_video_frame(
        _video_path(data_dir, episode_index, "observation.images.wrist_image"), frame_index
    )
    actions = np.asarray(table["action"], dtype=np.float32)
    gt_actions = _action_chunk(actions, row, horizon)

    obs = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.asarray(table["observation.state"][row], dtype=np.float32),
        "prompt": prompt,
    }
    return Sample(
        episode_index=episode_index,
        frame_index=frame_index,
        dataset_index=int(table["index"][row]),
        task_index=task_index,
        prompt=prompt,
        obs=obs,
        gt_actions=gt_actions,
    )


def _iter_default_samples(data_dir: pathlib.Path, tasks: dict[int, str], horizon: int, num_samples: int) -> list[Sample]:
    samples: list[Sample] = []
    for episode_file in sorted((data_dir / "data/chunk-000").glob("episode_*.parquet")):
        episode_index = int(episode_file.stem.split("_")[-1])
        table = pq.read_table(episode_file, columns=["frame_index"]).to_pydict()
        frame_indices = list(table["frame_index"])
        if len(frame_indices) < horizon:
            continue
        # Avoid the first frame and the tail to make GT chunks less edge-biased.
        candidate_rows = [0, min(5, len(frame_indices) - horizon), max(0, len(frame_indices) // 2)]
        for row in dict.fromkeys(candidate_rows):
            if row + horizon <= len(frame_indices):
                samples.append(_sample_from_episode(data_dir, tasks, episode_index, int(frame_indices[row]), horizon))
                if len(samples) >= num_samples:
                    return samples
    return samples


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _print_array_head(name: str, value: np.ndarray, rows: int) -> None:
    shown = np.asarray(value[:rows], dtype=np.float32)
    print(f"{name} first {len(shown)} actions:")
    print(np.array2string(shown, precision=6, suppress_small=False))


def _with_model_dtype(train_config: train_config_lib.TrainConfig, dtype: str) -> train_config_lib.TrainConfig:
    model_updates: dict[str, Any] = {"pytorch_compile_mode": None}
    if dtype == "bf16":
        return dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, **model_updates))
    if dtype == "fp32":
        model_updates["dtype"] = "float32"
        return dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, **model_updates))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _build_policy(
    config_name: str,
    checkpoint_dir: pathlib.Path,
    device: str | None,
    num_steps: int,
    dtype: str,
    tokenizer_path: pathlib.Path | None,
    allow_tf32: bool,
):
    _configure_torch_tf32(allow_tf32=allow_tf32)
    if tokenizer_path is not None:
        _install_tokenizer_override(tokenizer_path)
    train_config = train_config_lib.get_config(config_name)
    train_config = _with_model_dtype(train_config, dtype)
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.exists()

    if is_pytorch:
        start = time.monotonic()
        print("  constructing PI0Pytorch...", flush=True)
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        # PI0Pytorch.__init__ sets matmul precision to "high"; restore the
        # script-level setting afterwards so fp32 alignment tests do not use TF32.
        _configure_torch_tf32(allow_tf32=allow_tf32)
        print(f"  constructed in {time.monotonic() - start:.1f}s; loading safetensors...", flush=True)
        start = time.monotonic()
        safetensors.torch.load_model(model, str(weight_path))
        print(f"  safetensors loaded in {time.monotonic() - start:.1f}s", flush=True)
        if dtype == "bf16":
            start = time.monotonic()
            print("  converting selected torch params to bf16...", flush=True)
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
            print(f"  bf16 conversion done in {time.monotonic() - start:.1f}s", flush=True)
    else:
        restore_dtype = jnp.bfloat16 if dtype == "bf16" else jnp.float32
        start = time.monotonic()
        print(f"  restoring JAX params as {restore_dtype}...", flush=True)
        model = train_config.model.load(
            openpi_model.restore_params(checkpoint_dir / "params", dtype=restore_dtype)
        )
        print(f"  JAX params restored in {time.monotonic() - start:.1f}s", flush=True)

    start = time.monotonic()
    print("  creating data config/transforms...", flush=True)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    print(f"  data config ready in {time.monotonic() - start:.1f}s", flush=True)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    start = time.monotonic()
    print("  loading norm stats...", flush=True)
    norm_stats = openpi_checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    print(f"  norm stats loaded in {time.monotonic() - start:.1f}s", flush=True)

    if is_pytorch and device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.monotonic()
    print(f"  creating Policy wrapper (device={device if is_pytorch else 'jax'})...", flush=True)
    policy = openpi_policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=device if is_pytorch else None,
    )
    print(f"  Policy wrapper ready in {time.monotonic() - start:.1f}s", flush=True)
    return policy


def _install_tokenizer_override(tokenizer_path: pathlib.Path) -> None:
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    original_maybe_download = openpi_tokenizer.download.maybe_download

    def maybe_download(url: str, *args, **kwargs):
        if url == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer_path
        return original_maybe_download(url, *args, **kwargs)

    openpi_tokenizer.download.maybe_download = maybe_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-root", type=pathlib.Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--torch-ckpt", type=pathlib.Path, default=None)
    parser.add_argument("--jax-ckpt", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--allow-tf32", action="store_true", help="Allow TF32 matmul/convolution in PyTorch.")
    parser.add_argument("--tokenizer-path", type=pathlib.Path, default=None)
    parser.add_argument("--print-actions", type=int, default=3)
    args = parser.parse_args()
    _configure_torch_tf32(allow_tf32=args.allow_tf32)
    _print_torch_tf32_status()

    torch_ckpt = args.torch_ckpt or args.ckpt_root / "torch/pi05_libero"
    jax_ckpt = args.jax_ckpt or args.ckpt_root / "jax/pi05_libero"
    train_config = train_config_lib.get_config(args.config_name)
    horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim

    tasks = _load_tasks(args.data_dir)
    if args.episode_index is not None:
        samples = [
            _sample_from_episode(args.data_dir, tasks, args.episode_index, args.frame_index, horizon)
            for _ in range(args.num_samples)
        ]
    else:
        samples = _iter_default_samples(args.data_dir, tasks, horizon, args.num_samples)
    if not samples:
        raise RuntimeError(f"No valid samples found in {args.data_dir}")

    print(f"Loading torch policy: {torch_ckpt} dtype={args.dtype}")
    torch_policy = _build_policy(
        args.config_name,
        torch_ckpt,
        args.device,
        args.num_steps,
        args.dtype,
        args.tokenizer_path,
        args.allow_tf32,
    )
    print(f"Loading jax policy: {jax_ckpt} dtype={args.dtype}")
    jax_policy = _build_policy(
        args.config_name,
        jax_ckpt,
        None,
        args.num_steps,
        args.dtype,
        args.tokenizer_path,
        args.allow_tf32,
    )

    rng = np.random.default_rng(args.noise_seed)
    aggregate: dict[str, list[dict[str, float]]] = {"gt_torch": [], "gt_jax": [], "torch_jax": []}

    for sample_id, sample in enumerate(samples):
        noise = rng.standard_normal((horizon, action_dim), dtype=np.float32)
        torch_actions = np.asarray(torch_policy.infer(sample.obs, noise=noise)["actions"], dtype=np.float32)
        jax_actions = np.asarray(jax_policy.infer(sample.obs, noise=noise)["actions"], dtype=np.float32)
        gt_actions = sample.gt_actions[:, : torch_actions.shape[-1]]

        m_gt_torch = _metrics(gt_actions, torch_actions)
        m_gt_jax = _metrics(gt_actions, jax_actions)
        m_torch_jax = _metrics(torch_actions, jax_actions)
        aggregate["gt_torch"].append(m_gt_torch)
        aggregate["gt_jax"].append(m_gt_jax)
        aggregate["torch_jax"].append(m_torch_jax)

        print()
        print(
            f"sample={sample_id} episode={sample.episode_index} frame={sample.frame_index} "
            f"dataset_index={sample.dataset_index} task={sample.task_index}"
        )
        print(f"prompt: {sample.prompt}")
        print(f"GT    vs torch: {m_gt_torch}")
        print(f"GT    vs jax:   {m_gt_jax}")
        print(f"torch vs jax:   {m_torch_jax}")
        if args.print_actions > 0:
            _print_array_head("GT", gt_actions, args.print_actions)
            _print_array_head("torch", torch_actions, args.print_actions)
            _print_array_head("jax", jax_actions, args.print_actions)

    print()
    print("aggregate:")
    for key, values in aggregate.items():
        print(f"{key}: {_metrics_summary(values)}")


def _metrics_summary(values: list[dict[str, float]]) -> dict[str, float]:
    keys = values[0].keys()
    return {key: float(np.mean([item[key] for item in values])) for key in keys}


if __name__ == "__main__":
    main()
