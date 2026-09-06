"""Phase 3 (docs/plan.md) 3.1/3.2 -- trains the Stage A net (`nnue.py`'s `768 -> 512 -> 32 -> 1`)
in PyTorch, offline, then quantizes it to the int16 fixed-point scheme `nnue.nnue_forward`
expects and writes `weights/nnue.npz`.

Mirrors `nnue.py`'s architecture exactly so the quantized export is a faithful copy of what was
trained, not an approximation of it:
  - Layer 1 is an `EmbeddingBag(768, 512, mode="sum")` plus a bias -- summing one embedding row
    per active feature is exactly `nnue.py`'s "accumulator += w1[f, :] for each active f", just
    running in float32 during training instead of int64 at inference.
  - Layers 2 and 3 are plain `Linear`, ReLU between all three, no activation on the output --
    the same shape `nnue.nnue_forward` computes.

Per-layer weight scales (`nnue.W1_SCALE` etc.) come from `nnue.py` itself, so retuning them there
automatically retunes this script -- `ReLU(x / s) == ReLU(x) / s` for `s > 0` is what makes
training in real arithmetic and inferring in fixed-point integer arithmetic land on the same
function; see `nnue.py`'s module docstring for the full argument.

`torch`/CPU only -- never imported by `agent.py`/`evaluate.py`/`nnue.py`, so it costs nothing at
init time and ships nowhere (`tools/` is not packaged, see docs/plan.md 3.0).

Usage: .venv/Scripts/python.exe tools/nnue_train.py <dataset.npz> [<dataset2.npz> ...]
       [--epochs N] [--lr LR] [--clip-cp CP] [--out weights/nnue.npz]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn

from nnue import (
    HIDDEN1,
    HIDDEN2,
    INPUT_FEATURES,
    W1_SCALE,
    W2_SCALE,
    W3_SCALE,
    save_weights,
)


class StageA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.EmbeddingBag(INPUT_FEATURES, HIDDEN1, mode="sum")
        self.b1 = nn.Parameter(torch.zeros(HIDDEN1))
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, 1)
        nn.init.uniform_(self.w1.weight, -0.05, 0.05)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        a1 = torch.relu(self.w1(indices, offsets) + self.b1)
        a2 = torch.relu(self.fc2(a1))
        return self.fc3(a2).squeeze(-1)  # type: ignore[no-any-return]


def load_dataset(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for path in paths:
        data = np.load(path)
        features.append(data["features"])
        labels.append(data["labels"])
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def to_embeddingbag_input(feature_matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattens the -1-padded `[N, MAX_ACTIVE_FEATURES]` matrix into EmbeddingBag's
    `(concatenated indices, per-sample offsets)` form, dropping the padding sentinels.
    """
    offsets = [0]
    flat: list[int] = []
    for row in feature_matrix:
        active = row[row >= 0]
        flat.extend(int(x) for x in active)
        offsets.append(len(flat))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)


def quantize_and_save(model: StageA, out_path: str) -> None:
    w1 = model.w1.weight.detach().numpy() * W1_SCALE
    b1 = model.b1.detach().numpy() * W1_SCALE
    w2 = model.fc2.weight.detach().numpy().T * W2_SCALE
    b2 = model.fc2.bias.detach().numpy() * W2_SCALE
    w3 = model.fc3.weight.detach().numpy().reshape(-1) * W3_SCALE
    b3 = model.fc3.bias.detach().numpy() * W3_SCALE

    def q(arr: np.ndarray, dtype: type) -> np.ndarray:
        info = np.iinfo(dtype)
        return np.clip(np.round(arr), info.min, info.max).astype(dtype)

    save_weights(
        out_path,
        q(w1, np.int16), q(b1, np.int32),
        q(w2, np.int16), q(b2, np.int32),
        q(w3, np.int16), q(b3, np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-cp", type=float, default=1000.0)
    parser.add_argument("--out", default="weights/nnue.npz")
    args = parser.parse_args()

    feature_matrix, labels = load_dataset(args.datasets)
    labels = np.clip(labels, -args.clip_cp, args.clip_cp).astype(np.float32)
    indices, offsets = to_embeddingbag_input(feature_matrix)
    targets = torch.tensor(labels, dtype=torch.float32) / args.clip_cp

    model = StageA()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(indices, offsets) / args.clip_cp
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs}: mse={loss.item():.6f}", flush=True)

    quantize_and_save(model, args.out)
    print(f"\nwrote quantized Stage A weights to {args.out}")


if __name__ == "__main__":
    main()
