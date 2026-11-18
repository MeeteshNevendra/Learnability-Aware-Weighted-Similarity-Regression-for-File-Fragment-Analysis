# Data preparation

| Script | Input | Output |
|---|---|---|
| `download_kaggle.py --out DIR` | the public Kaggle release | `train.npz`, `val.npz`, `test.npz`, `*_meta.csv` in `DIR` |
| `prepare_kaggle.py --kaggle-dir DIR --out ROOT` | the extracted release | the dataset folder read by `experiments/` (see below) |
| `build_govdocs1.py ROOT GOVDOCS1_DIR` | the extracted members of Govdocs1 archive `001.zip` | `ROOT/08_govdocs1_external/` |

The folder written by `prepare_kaggle.py`:

```
ROOT/
  dataset_info.json                 source, similarity and target settings, counts
  02_splits/file_split.csv          file id, extension, partition, recovered source length, retained fragments
  03_fragments/fragments.csv        one row per retained fragment (+ fragment_bytes.npy)
  04_pairs/{train,val,test}.npz     X1, X2 (bytes), y (Eq. 23), z, mode, S (Eqs. 7-11)
  04_pairs/{train,val,test}_pairs.csv   the same pairs with offsets, positions, both targets and both
                                    versions of the similarities (recomputed and released_*)
  05_retrieval/                     test candidate pool, eligible successor queries, tie order (seed 42)
  06_type_cue/test_label_corruption.csv   nested corruption selections of the test files
  07_png_repair/                    only with --png-sources (intact test PNGs named <file_id>.png)
  reports/conversion_report.txt     every check made during the conversion
```

`build_govdocs1.py` adds `08_govdocs1_external/`. It contains seven conditions, each with:
- 20,000 pairs (5,000 per sampling mode), with the Eq. 23 reference recomputed in the condition's stream;
- a 76-file retrieval pool (5,047 blocks, 4,971 queries, 5,046 candidates per query);
- the per-file stream parameters (start offsets, gzip level, AES-256 keys and counter blocks).

The seven conditions are aligned 512, 256, 1,024 and 4,096-byte blocks; unaligned 512-byte blocks; gzip; and AES-256-CTR. The seeds are fixed, so the output is deterministic.
