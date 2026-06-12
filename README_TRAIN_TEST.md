# OpenPI LIBERO 训练与测试说明

本文档记录当前机器上已经验证过的 OpenPI + LeRobot 0.4.3 + LIBERO 本地数据训练流程。

注意：OpenPI 的 uv 环境用于 `appstart` 进入后的 Apptainer 类容器环境。训练和测试命令建议先进入容器，再激活 uv 环境；不要在裸 CentOS 7 shell 中直接跑训练。

## 0. 进入 Apptainer 环境

本项目训练/测试应在 `appstart` 启动后的 Apptainer 类容器环境中运行。先进入容器：

```bash
appstart
```

进入容器后，再执行下面的环境激活和训练命令。

## 1. 环境

项目目录：

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
```

Python 环境由 `uv` 创建，目录为：

```text
/mnt/petrelfs/zhuyangkun/envs/uv/openpi
```

日常使用建议直接 `source` 这个 venv：

```bash
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate
```

不激活环境时，也可以显式使用该环境的 Python：

```bash
/mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/python scripts/train.py ...
```

如果要用 `uv` 启动，指定同一个 Python 环境：

```bash
uv run --python /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/python python scripts/train.py ...
```

已验证关键版本：

```text
openpi        0.1.0
openpi-client 0.1.0
lerobot       0.4.3
torch         2.7.1
jax           0.5.3
numpy         2.0.2
```

注意：上面的训练命令是在 `appstart` 容器环境中验证的。当前 uv 环境里的 NumPy 版本为 `2.0.2`，满足 LeRobot 0.4.3 的 `numpy>=2` 要求。

本地资源路径：

```text
LIBERO 数据:
/mnt/inspurfs/wam_agent/share_data_checkpoint/libero

OpenPI 模型与 tokenizer:
/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi

pi05 base checkpoint:
/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi/openpi-assets/checkpoints/pi05_base/params
```

建议每次训练前设置：

```bash
export OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

## 2. 当前可用配置

当前本地训练配置名：

```text
pi05_libero_local
```

该配置使用：

- 模型：`pi05`
- 数据：本地 LIBERO LeRobot 数据
- checkpoint：本地 `pi05_base/params`
- FSDP：`fsdp_devices=8`
- batch size：`8`
- wandb：命令行中建议关闭

注意：`pi05` 全量训练必须使用 FSDP。之前 `fsdp_devices=1` 会 OOM；`fsdp_devices=8` 已验证可以跑通。

## 3. 计算 Normalization Stats

训练前需要有 OpenPI 格式的 normalization stats。

快速连通性版本，使用 512 帧：

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate

OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
python scripts/compute_norm_stats.py \
  --config-name pi05_libero_local \
  --max-frames 512
```

输出目录：

```text
assets/pi05_libero_local/physical-intelligence/libero
```

正式训练建议使用全量数据重新计算：

```bash
OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
python scripts/compute_norm_stats.py \
  --config-name pi05_libero_local
```

## 4. 1-Step 训练测试

用于确认环境、数据、checkpoint、FSDP 和训练 step 都能跑通：

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate

OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi05_libero_local \
  --exp-name=pi05_newenv_1step \
  --overwrite \
  --num-train-steps=1 \
  --log-interval=1 \
  --save-interval=1 \
  --keep-period=None \
  --no-wandb-enabled
```

已验证输出：

```text
Step 0: grad_norm=1.6173, loss=0.1148, param_norm=1802.3864
```

checkpoint 输出：

```text
checkpoints/pi05_libero_local/pi05_newenv_1step/0
```

该 1-step checkpoint 约 41G。

## 5. 正式训练

确认 1-step 正常后，可以去掉测试步数限制：

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate

OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi05_libero_local \
  --exp-name=pi05_libero_full \
  --overwrite \
  --no-wandb-enabled
```

如需开启 wandb，去掉 `--no-wandb-enabled`。

如果要从已有实验继续训练：

```bash
OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi05_libero_local \
  --exp-name=pi05_libero_full \
  --resume \
  --no-wandb-enabled
```

## 6. 简单推理/服务测试

训练完成后可以用 checkpoint 启动 policy server：

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate

OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero_local \
  --policy.dir=checkpoints/pi05_libero_local/pi05_newenv_1step/0
```

默认监听端口是 `8000`。

如果要测试正式训练的 checkpoint，把 `--policy.dir` 改成对应 step：

```text
checkpoints/pi05_libero_local/<exp-name>/<step>
```

## 7. 常见问题

### 7.1 `Normalization stats not found`

先运行：

```bash
OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
python scripts/compute_norm_stats.py --config-name pi05_libero_local
```

### 7.2 tokenizer 下载卡住

必须设置：

```bash
export OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi
```

该目录下已有：

```text
big_vision/paligemma_tokenizer.model
```

### 7.3 full pi05 OOM

确认配置使用了：

```text
fsdp_devices=8
```

`fsdp_devices=1` 会让每张卡复制完整模型，已验证会 OOM。

### 7.4 LeRobot 0.4.3 兼容

当前代码已做兼容：

- `lerobot.common.datasets...` fallback 到 `lerobot.datasets...`
- 支持本地 `repo_root`
- 兼容 LeRobot 0.4.3 的 tasks DataFrame
- `download_videos=False`

## 8. 当前已验证命令摘要

```bash
cd /mnt/petrelfs/zhuyangkun/workspace/pi_05/openpi
source /mnt/petrelfs/zhuyangkun/envs/uv/openpi/bin/activate

OPENPI_DATA_HOME=/mnt/inspurfs/wam_agent/share_data_checkpoint/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi05_libero_local \
  --exp-name=pi05_newenv_1step \
  --overwrite \
  --num-train-steps=1 \
  --log-interval=1 \
  --save-interval=1 \
  --keep-period=None \
  --no-wandb-enabled
```
