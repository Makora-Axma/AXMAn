#!/usr/bin/env python3
"""
AXMAN (AXMANet) Autonomous Pre-Training & Fine-Tuning CLI.
Part of AXMAN (AXMANet) Phase 6.2 Implementation.

Usage:
  python train.py --config small --epochs 5 --batch-size 8
  python train.py --dataset path/to/proofs.axmb --val-dataset path/to/val.axmb
  python train.py --balancer gradnorm --mixed-precision bf16
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional

import numpy as np
import torch

# Ensure axman package is resolvable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from axman.core.config import AxmanConfig
from axman.core.data import (
    HypergraphSample,
    HypergraphDataset,
    TopologicalAugmentor,
    create_hypergraph_dataloader,
)
from axman.core.heads.policy_head import TacticConstructorKind
from axman.core.training import (
    AXMANetModel,
    AXMANetTrainer,
    TrainingConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AXMANet-Train")


def generate_synthetic_samples(
    num_samples: int = 64,
    feat_dim: int = 256,
    seed: int = 42,
) -> List[HypergraphSample]:
    """
    Generate synthetic proof hypergraphs for self-contained autonomous training.
    """
    rng = np.random.RandomState(seed)
    samples: List[HypergraphSample] = []

    for i in range(num_samples):
        num_nodes = rng.randint(4, 16)
        num_edges = rng.randint(2, max(num_nodes // 2, 3))

        node_features = rng.randn(num_nodes, feat_dim).astype(np.float32)

        sources = []
        ports = []
        targets = []
        offsets = [0]

        for _ in range(num_edges):
            arity = rng.randint(1, min(num_nodes, 4) + 1)
            src_nodes = rng.choice(num_nodes, size=arity, replace=False)
            tgt_node = rng.choice(num_nodes)

            sources.extend(src_nodes)
            ports.extend(list(range(arity)))
            targets.append(tgt_node)
            offsets.append(offsets[-1] + arity)

        target_action = rng.randint(0, len(TacticConstructorKind))
        target_prems = rng.choice(num_nodes, size=min(2, num_nodes), replace=False).tolist()
        target_val = float(rng.rand())
        target_dist = float(rng.randint(1, 10))
        target_lat = rng.randn(feat_dim).astype(np.float32)
        open_goals = (rng.rand(num_nodes) > 0.5).astype(bool)

        samples.append(
            HypergraphSample(
                node_features=node_features,
                edge_sources=np.array(sources, dtype=np.int64),
                edge_ports=np.array(ports, dtype=np.int64),
                edge_targets=np.array(targets, dtype=np.int64),
                edge_offsets=np.array(offsets, dtype=np.int64),
                target_action=target_action,
                target_premises=target_prems,
                target_value=target_val,
                target_distance=target_dist,
                target_latent=target_lat,
                open_goal_mask=open_goals,
                metadata={"synthetic_id": i},
            )
        )

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AXMANet Core Architecture")

    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Architecture preset (small, base, large)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to training dataset (.axmb, .arrow)",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="",
        help="Path to validation dataset (.axmb, .arrow)",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=64,
        help="Number of synthetic samples to generate if no dataset is provided",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate for backbone and policy",
    )
    parser.add_argument(
        "--lr-flow",
        type=float,
        default=2e-4,
        help="Learning rate for diffusion flow matching head",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--balancer",
        type=str,
        default="gradnorm",
        choices=["gradnorm", "pcgrad", "static"],
        help="Multi-task loss balancing strategy",
    )
    parser.add_argument(
        "--gradnorm-alpha",
        type=float,
        default=0.16,
        help="GradNorm alpha hyperparameter",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints and safetensors files",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Initializing AXMANet training on device: %s", device)
    logger.info("Configuration preset: %s | Balancer: %s", args.config, args.balancer)

    # 1. Model Configuration
    if args.config == "small":
        model_config = AxmanConfig.small()
    elif args.config == "base":
        model_config = AxmanConfig.base()
    else:
        model_config = AxmanConfig.large()

    # 2. Build Dataset
    augmentor = TopologicalAugmentor(
        p_permute=0.5,
        p_jitter=0.3,
        jitter_std=0.01,
        p_edge_dropout=0.1,
    )

    if args.dataset and Path(args.dataset).exists():
        logger.info("Loading training dataset from: %s", args.dataset)
        train_ds = HypergraphDataset(args.dataset, transform=augmentor)
    else:
        logger.info("Generating %d synthetic training hypergraphs...", args.synthetic_samples)
        samples = generate_synthetic_samples(
            num_samples=args.synthetic_samples,
            feat_dim=model_config.d_model,
        )
        train_ds = HypergraphDataset(samples, transform=augmentor)

    train_loader = create_hypergraph_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = None
    if args.val_dataset and Path(args.val_dataset).exists():
        val_ds = HypergraphDataset(args.val_dataset)
        val_loader = create_hypergraph_dataloader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # 3. Instantiate Model
    model = AXMANetModel(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameter count: %s (%.2f M)", f"{num_params:,}", num_params / 1e6)

    # 4. Trainer Configuration
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr,
        lr_policy=args.lr,
        lr_value=args.lr,
        lr_flow=args.lr_flow,
        weight_decay=args.weight_decay,
        balancer_type=args.balancer,
        gradnorm_alpha=args.gradnorm_alpha,
        mixed_precision=args.mixed_precision,
        device=device,
        output_dir=args.output_dir,
    )

    trainer = AXMANetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_cfg,
    )

    # 5. Execute Training Loop
    logger.info("Starting training loop for %d epochs...", args.epochs)
    history = trainer.train()

    logger.info("Training finished successfully across %d epochs!", len(history))
    for m in history:
        weights_str = ""
        if m.task_weights:
            weights_str = " | Weights: " + ", ".join(f"{k}={v:.2f}" for k, v in m.task_weights.items())
        logger.info(
            "Epoch %d/%d: Total Loss=%.4f (Policy=%.4f, Premise=%.4f, Value=%.4f, Flow=%.4f) | Acc=%.2f%%%s (%.2fs)",
            m.epoch,
            args.epochs,
            m.train_total_loss,
            m.train_policy_loss,
            m.train_premise_loss,
            m.train_value_loss,
            m.train_flow_loss,
            m.train_policy_acc * 100.0,
            weights_str,
            m.elapsed_seconds,
        )

    logger.info("Checkpoints and Safetensors saved in: %s", Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
