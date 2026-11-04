"""Train one raw-byte Siamese regression model on the frozen replacement collection.

Protocol (archived revision/train.py, unchanged): AdamW lr 8e-4, weight decay 1e-4, batch 1024 pairs, CUDA autocast
(float16) + GradScaler, at most 20 epochs, ReduceLROnPlateau(mode=min, factor=0.5, patience=2) on validation loss
(PyTorch halves the rate when the number of consecutive non-improving epochs exceeds 2, i.e. at the third),
early stop after 5 consecutive epochs without a validation-loss improvement of more than 1e-5, keep the
lowest-validation-loss checkpoint. Validation loss uses a fixed generator (seed 0) for the ranking term.
Inputs are raw bytes / 255 only; component values and metadata are used only to build the target.

Differences from the archived script (documented): data are read from data/fragments + data/pairs; an epoch-level
resume checkpoint (last.pt) is written; the test split is NOT loaded here (test evaluation is a separate, later step
that requires configs/frozen_protocol.json).

usage: python scripts/30_train.py --name heuristic_s42 --weights '{"s_hist":0.25,...}' --seed 42 [--exclude-ext heic]
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import components as C
import model as M
from paths import CONFIGS, FRAGMENTS, PAIRS, RUNS

EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY = 20, 5, 1024, 8e-4, 1e-4
MIN_DELTA = 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_weights(w):
    w = {k: float(v) for k, v in w.items() if float(v) > 0}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def regression_metrics(y, p):
    from scipy.stats import spearmanr
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    mse = float(np.mean((y - p) ** 2))
    var = float(np.var(y))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(y - p))),
            "r2": 1 - mse / var if var > 0 else float("nan"),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan"),
            "spearman": float(spearmanr(y, p).correlation)}


class SplitData:
    def __init__(self, split, weights, device, exclude_ext=None):
        frags = np.load(os.path.join(FRAGMENTS, f"{split}_frags.npy"))
        z = np.load(os.path.join(PAIRS, f"{split}_pairs.npz"))
        a, b = z["a"].astype(np.int64), z["b"].astype(np.int64)
        y = C.composite({k: z[k] for k in C.COMPONENTS}, weights).astype(np.float32)
        keep = np.ones(len(a), dtype=bool)
        if exclude_ext:
            ext = pd.read_csv(os.path.join(FRAGMENTS, f"{split}_fragments.csv"), usecols=["ext"]).ext.to_numpy()
            keep = (ext[a] != exclude_ext) & (ext[b] != exclude_ext)
        self.keep = keep
        self.y_np = y[keep]
        self.F = torch.from_numpy(frags).to(device)
        self.a = torch.from_numpy(a[keep]).to(device)
        self.b = torch.from_numpy(b[keep]).to(device)
        self.y = torch.from_numpy(self.y_np).to(device)

    def __len__(self):
        return len(self.y)

    def batches(self, bs, shuffle=False, generator=None):
        n = len(self)
        order = torch.randperm(n, device=self.y.device, generator=generator) if shuffle else torch.arange(n, device=self.y.device)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            yield self.F[self.a[idx]].float() / 255.0, self.F[self.b[idx]].float() / 255.0, self.y[idx]


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    preds, total = [], 0.0
    for x1, x2, y in data.batches(BATCH_SIZE):
        with torch.amp.autocast("cuda"):
            _, _, pred, cos = model(x1, x2)
            loss = M.combined_loss(pred.float(), cos.float(), y, generator=gen)
        total += loss.item() * len(y)
        preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds), total / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-ext")
    ap.add_argument("--max-epochs", type=int, default=EPOCHS, help="only for the pipeline check")
    ap.add_argument("--max-train-pairs", type=int, default=0, help="only for the pipeline check")
    args = ap.parse_args()

    weights = normalize_weights(json.loads(args.weights))
    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "summary_val.json")):
        print(f"[{args.name}] already complete", flush=True)
        return
    freeze = json.load(open(os.path.join(CONFIGS, "collection_freeze.json")))
    config = {"name": args.name, "weights": weights, "seed": args.seed, "exclude_ext": args.exclude_ext,
              "epochs": args.max_epochs, "patience": PATIENCE, "batch_size": BATCH_SIZE, "lr": LR,
              "weight_decay": WEIGHT_DECAY, "scheduler": "ReduceLROnPlateau(min,0.5,patience=2)",
              "amp": "float16 autocast + GradScaler", "collection_digest": freeze["accepted_files_sha256_digest"],
              "activation_checkpointing": "transformer layers (execution detail; numerically equivalent)",
              "max_train_pairs": args.max_train_pairs}
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    train = SplitData("train", weights, device, args.exclude_ext)
    val = SplitData("val", weights, device, args.exclude_ext)
    if args.max_train_pairs:
        for k in ("a", "b", "y"):
            setattr(train, k, getattr(train, k)[:args.max_train_pairs])
    print(f"[{args.name}] weights={weights} seed={args.seed} train={len(train)} val={len(val)} "
          f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    model = M.ContinuousSimilarityModel().to(device)
    # Execution detail on the shared GPU (not a protocol change): transformer layers are recomputed in the backward
    # pass (activation checkpointing). Same batch of 1024, same BatchNorm updates, same loss; gradients equal the
    # plain path to <= 5.7e-10 in float32 (logs/benchmark_step.json); peak memory 6.6 GB instead of 16 GB.
    model.set_activation_checkpointing("layer")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best, bad, history, start_epoch = float("inf"), 0, [], 1
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        best, bad, history, start_epoch = st["best"], st["bad"], st["history"], st["epoch"] + 1
        print(f"[{args.name}] resumed after epoch {st['epoch']}", flush=True)

    t_start = time.time()
    stopped = bad >= PATIENCE
    for epoch in range(start_epoch, args.max_epochs + 1):
        if stopped:
            break
        model.train()
        t0, tl = time.time(), 0.0
        for x1, x2, y in train.batches(BATCH_SIZE, shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                _, _, pred, cos = model(x1, x2)
                loss = M.combined_loss(pred.float(), cos.float(), y, generator=gen)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
        val_pred, val_loss = predict(model, val, device)
        sched.step(val_loss)
        vm = regression_metrics(val.y_np, val_pred)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tl / len(train), "val_loss": val_loss,
               **{f"val_{k}": v for k, v in vm.items()}, "seconds": time.time() - t0}
        history.append(row)
        print(f"[{args.name}] epoch {epoch} train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} "
              f"val_r2={vm['r2']:.4f} lr={row['lr']:.2e} ({row['seconds']:.0f}s)", flush=True)
        if val_loss < best - MIN_DELTA:
            best, bad = val_loss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss,
                        "weights": weights, "seed": args.seed, "exclude_ext": args.exclude_ext,
                        "config_sha256": config["config_sha256"]}, os.path.join(out, "best_model.pt"))
            np.save(os.path.join(out, "val_pred_best.npy"), val_pred.astype(np.float32))
        else:
            bad += 1
            stopped = bad >= PATIENCE
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                    "best": best, "bad": bad, "history": history, "epoch": epoch}, last)

    with open(os.path.join(out, "training_history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    ck = torch.load(os.path.join(out, "best_model.pt"), map_location="cpu", weights_only=False)
    best_row = history[ck["epoch"] - 1]
    summary = {"name": args.name, "weights": weights, "seed": args.seed, "exclude_ext": args.exclude_ext,
               "best_epoch": ck["epoch"], "epochs_run": len(history), "best_val_loss": ck["val_loss"],
               "best_val_r2": best_row["val_r2"], "val_at_best": {k[4:]: v for k, v in best_row.items() if k.startswith("val_")},
               "train_minutes_this_session": (time.time() - t_start) / 60,
               "total_train_seconds": float(sum(r["seconds"] for r in history)),
               "n_parameters": M.n_parameters(model), "config_sha256": config["config_sha256"],
               "stopped_early": bool(stopped), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if args.max_train_pairs:
        summary["pipeline_check_only"] = True
    json.dump(summary, open(os.path.join(out, "summary_val.json"), "w"), indent=1)
    os.remove(last)
    print(f"[{args.name}] done best_epoch={ck['epoch']} best_val_r2={best_row['val_r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
