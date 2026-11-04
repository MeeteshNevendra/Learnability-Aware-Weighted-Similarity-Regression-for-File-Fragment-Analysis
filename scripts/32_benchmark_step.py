"""Training-step benchmark on the current (shared) GPU: step time and peak memory for batch 1024, with and without
activation checkpointing of the transformer layers, plus a numerical-equivalence check of the two execution paths.
Output: logs/benchmark_step.json"""
import json
import os
import time

import numpy as np
import torch

import model as M
from paths import FRAGMENTS, LOGS


def main():
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    F_ = torch.from_numpy(np.load(os.path.join(FRAGMENTS, "train_frags.npy"))).to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    out = {"gpu": torch.cuda.get_device_name(0), "free_gb_at_start": torch.cuda.mem_get_info()[0] / 2**30}
    # equivalence: same weights, same batch, same dropout seed -> same loss and gradients
    torch.manual_seed(0)
    base = M.ContinuousSimilarityModel().to(dev)
    idx = torch.randint(0, len(F_), (64,), device=dev, generator=g)
    jdx = torch.randint(0, len(F_), (64,), device=dev, generator=g)
    y = torch.rand(64, device=dev, generator=g)
    grads = {}
    MODES = (False, "layer", "ff", "sa", "both")
    out["sdpa"] = {"flash_available": bool(getattr(torch.backends.cuda, "is_flash_attention_available", lambda: None)()),
                   "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                   "mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled()}
    for ck in MODES:
        mdl = M.ContinuousSimilarityModel().to(dev)
        mdl.load_state_dict(base.state_dict())
        mdl.train()
        mdl.set_activation_checkpointing(ck)
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        _, _, p, c = mdl(F_[idx].float() / 255, F_[jdx].float() / 255)
        loss = M.combined_loss(p.float(), c.float(), y, generator=torch.Generator(device=dev).manual_seed(5))
        loss.backward()
        grads[ck] = (loss.item(), torch.cat([q.grad.flatten() for q in mdl.parameters() if q.grad is not None]))
    out["equivalence_fp32"] = {str(k): {"loss": grads[k][0], "max_abs_grad_diff_vs_plain":
                                        float((grads[False][1] - grads[k][1]).abs().max())} for k in MODES}
    out["equivalence_fp32"]["max_abs_grad"] = float(grads[False][1].abs().max())
    for ck in MODES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mdl = M.ContinuousSimilarityModel().to(dev)
        mdl.set_activation_checkpointing(ck)
        opt = torch.optim.AdamW(mdl.parameters(), lr=8e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        times = []
        for step in range(8):
            a = torch.randint(0, len(F_), (1024,), device=dev, generator=g)
            b = torch.randint(0, len(F_), (1024,), device=dev, generator=g)
            yy = torch.rand(1024, device=dev, generator=g)
            torch.cuda.synchronize()
            t = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                _, _, p, c = mdl(F_[a].float() / 255, F_[b].float() / 255)
                loss = M.combined_loss(p.float(), c.float(), yy, generator=g)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        out[f"checkpointing_{ck}"] = {"median_step_seconds": float(np.median(times[2:])),
                                      "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                                      "estimated_train_epoch_minutes": float(np.median(times[2:])) * 293 / 60}
        del mdl, opt
    json.dump(out, open(os.path.join(LOGS, "benchmark_step.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
