# Protocol deviations and execution details (replacement-collection runs)

1. Execution detail, not a protocol change (recorded 2026-10-27, before training): the GPU (RTX 4000 Ada, 20 GB) was
   shared with another user process whose memory use varied between about 0.4 and 11 GB. Plain batch-1024 training
   peaks at 16 GB and, when combined with the other process, overflowed into WDDM paging (70 s per step instead of
   1.2 s; logs/benchmark_step.json). All training runs therefore recompute the three transformer encoder layers in the
   backward pass (torch.utils.checkpoint, preserve_rng_state=True). Batch size (1024), BatchNorm statistics (CNN branch
   not recomputed), dropout masks, loss and optimiser are unchanged; float32 gradients match the plain path to
   <= 5.7e-10. Peak memory 6.6 GB.
2. Compression ratio uses classic zlib 1.3.1 (helper interpreter CPython 3.12.10) because the archived s_comp values
   were produced with classic zlib; CPython 3.14 links zlib-ng (audit/zlib_check.json).
3. Fragment extraction uses complete aligned 512-byte blocks only; the archived code appended one unaligned block at
   offset size-512 per file (audit/archive_audit.json: one such block per archived file).
4. Learning-rate schedule follows the archived code: ReduceLROnPlateau(factor=0.5, patience=2), which halves the rate
   at the third consecutive epoch without improvement (PyTorch semantics); the manuscript text says "after two".
5. Pair head scoring for provenance tasks keeps the archived float16 autocast and 0.5*(P+P^T) symmetrisation; a float32
   rescoring of the three main models is reported as a declared sensitivity analysis.
6. Execution incident (2026-10-28, after the freeze): `scripts/51_govdocs_eval.py` chose models by the existence of
   `best_model.pt`, which is rewritten while a run trains, and therefore scored the unfinished `heuristic_s43`
   checkpoint (epoch 4-5) on Govdocs1. Those outputs were moved unchanged to
   `results_invalid/govdocs1_heuristic_s43_unfinished_checkpoint/` and are not used. The script now also requires
   `summary_val.json` (logged amendment in `configs/protocol_amendments.log`); the finished checkpoint is evaluated
   after training. No setting, weight, metric or candidate rule changed, and no decision used the invalid outputs.
7. Addition after the freeze (2026-11-03, after all results were computed): paired bootstrap differences of the ten
   ablation and subset models and the s_hist and s_ngram component models against the six-component heuristic model
   (seed 42), with heuristic seeds 43 and 44 as a reference for seed-only differences (`scripts/61_ablation_paired.py`,
   `results/bootstrap/local_paired_vs_heuristic_s42.json`, table `T12b_ablation_paired_differences`). The frozen
   protocol defines paired differences against histogram cosine only. The new comparison reuses the stored replicates
   of the frozen local bootstrap (same file draws); no frozen script, setting, model or existing output changed.
