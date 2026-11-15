# RAVEL-FCL

Federated class-incremental learning under non-monotonic class streams.

This repository contains a reference implementation of RAVEL-FCL: a ReACGAN-based
federated continual learner that supports recurring (non-monotonic) class streams,
variational consolidation (EVCL), zCDP communication compression/privacy, an
MMD + dual-gate defense, and poisoning-attack simulation.

## Project structure

```
main.py                 CLI entry point
configs/                Reference hyperparameter files
data/                   Dataset loading and recurring class-stream generation
models/                 ReACGAN generator/discriminator
losses/                 D2D-CE, alignment, and adversarial losses
federated/              Client, server, compression, and defense
attacks/                Label-flip and stealthy poisoning attacks
utils/                  Metrics, seeding, and device helpers
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA optional)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run a small CPU smoke test:

```bash
python main.py --dataset mnist --num_clients 2 --num_tasks 2 \
    --rounds_per_task 2 --local_epochs 1 --batches_per_round 1 \
    --batch_size 8 --eval_every 1 --eval_max_samples 256 \
    --base_channels 64 --num_fim_batches 2 --evcl_steps 1 \
    --dp_epsilon inf --run_name smoke_mnist
```

Run the standard baseline (expensive):

```bash
python main.py --dataset emnist --num_clients 5 --num_tasks 5 \
    --rounds_per_task 40 --local_epochs 5
```

Run with the full Extension C defense and a stealthy attack:

```bash
python main.py --dataset cifar10 --num_clients 5 --num_tasks 5 \
    --rounds_per_task 40 --local_epochs 5 \
    --defense_mode full --attack_type stealthy --malicious_ratio 0.2
```

## Key options

| Area | Flags |
| --- | --- |
| Stream recurrence | `--recurrence_bias` / `--omega` |
| Partial participation | `--participation_rate` |
| Asynchronous tasks | `--async_offset_max`, `--transition_offsets`, `--random_task_lengths` |
| Privacy / compression | `--dp_epsilon`, `--keep_ratio`, `--l2_clip` |
| Defense | `--defense_mode none\|stage1\|stage2\|full` |
| Attacks | `--attack_type none\|label_flip\|stealthy`, `--malicious_ratio` |

Datasets (`mnist`, `emnist`, `cifar10`, `synthetic`) are downloaded on first use
into `data/raw/` unless cached. Run `python main.py --help` for the full CLI.

## Outputs

Each run writes `summary.json`, `metrics.csv`, and `metrics.json` to a
`results/<run_name>/` directory. `results/` is git-ignored and is regenerated
by running the code.

## Configuration files

The YAML files in `configs/` document the reference values from the technical
specification and stress-test grid. The CLI in `main.py` is the actual interface;
the YAML files are not loaded automatically.
