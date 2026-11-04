# Fragment similarity regression

Code for a study of learned similarity scores for file fragments (manuscript under review; the reference will be added
when the article is published). A Siamese network is trained to predict a weighted similarity score for pairs of
512-byte file fragments, with three ways of weighting the score's six components: heuristic, uniform and
learnability-aware. The study then tests whether the learned scores help with forensic provenance tasks (same-file
retrieval, ranking of adjacent blocks, clustering, chaining of consecutive blocks, and an external Govdocs1 test set)
compared with direct byte-statistic baselines.

This repository holds the complete pipeline as it was run: construction of the 500-file data set, fragments and pairs,
training, the frozen evaluation protocol, evaluation, bootstrap confidence intervals, tables and figures. The data set,
**FileFragment500 v1.0**, is published separately (DOI to be added): 500 files, 50 in each of ten recorded-extension
groups (PNG, JPG, HEIC, PDF, MP4, DOCX, PPTX, JPEG, WEBM and CSV), split 350/75/75 into training, validation and test
files.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | Pipeline scripts, numbered in the order in which they are run, and their shared modules |
| `tools/` | Data set tools: install, retrieve the NapierOne files, verify, build the fragment arrays, Linux/macOS path links |
| `configs/` | Frozen evaluation protocol, its amendments and deviations, collection freeze record, bigram hash table, weights of the three targets |
| `environment/` | Full package list and hardware of the original run |
| `results/tables/` | Result tables of the study (CSV and Markdown), as written by `scripts/70_tables.py` |
| `data/` | Empty; the data set goes here (`data/README.md`) |
| `requirements.txt` | Direct dependencies, pinned to the tested versions |

| Stage | Scripts |
|---|---|
| Data set construction | `fetch_archives.py` (source archives), `10_enumerate_commons.py` and `10b_extend_commons_probes.py` (Wikimedia Commons candidates), `11_collect_zenodo.py` (Zenodo HEIC candidates), `12_verify_napierone_archives.py`, `13_select_and_verify.py` (selection rules R1-R4), `14_families_and_split.py` (content families, split, freeze record), `15_check_collection.py`, `20_build_fragments.py` (512-byte fragments and byte statistics), `21_build_pairs.py` (pairs and similarity components), `50_govdocs_build.py` (Govdocs1 test set) |
| Training | `30_train.py` (one run), `31_queue.py` (all runs, resumable), `32_benchmark_step.py` (speed and memory check) |
| Evaluation | `40_freeze_protocol.py` (frozen protocol and amendments), `41_evaluate_local.py` (regression, provenance, shortlist, reconstruction, leave-one-type-out, cost), `51_govdocs_eval.py`, `60_bootstrap.py`, `61_ablation_paired.py` |
| Reporting | `70_tables.py`, `71_figures.py`, `72_submission_figures.py` |
| Shared modules | `paths.py` (folders and constants), `formats.py` (format verification), `components.py` (similarity components, byte statistics), `model.py` (network), `evallib.py` (evaluation), `bootlib.py` (bootstrap), `zlib_classic_lengths.py` (classic zlib helper) |
| Windows runner | `finish_all.cmd` |

## Requirements

* **Python.** Tested with CPython 3.14.5 on Windows 11. Other recent versions may work with suitable package versions.
* **GPU.** Training and evaluation used an NVIDIA RTX 4000 Ada (20 GB) with CUDA 13.0. The 26 training runs
  took about 45 hours in total (0.9 to 2.6 hours each, on a GPU shared with other work).
* **Packages.** `requirements.txt` lists the direct dependencies; `environment/requirements-lock.txt` is the full
  package list of the original run.

```
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux or macOS: source .venv/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130   # or the build for your CUDA version
pip install -r requirements.txt
```

## 1. Data

Download `FileFragment500_v1.0.zip` (DOI to be added), then run from the repository root:

```
python tools/install_dataset.py path/to/FileFragment500_v1.0.zip   # copies its data/ folder here, checking every file
python tools/retrieve_napierone.py      # downloads the 250 NapierOne files (about 603 MB) and checks every file
python tools/verify_collection.py       # all 500 files match the frozen collection
python tools/build_fragment_arrays.py   # writes data/fragments/{train,val,test}_frags.npy
python tools/posix_stored_paths.py      # Linux and macOS only (see Notes)
```

The data set holds the source files (250 of them; the NapierOne files are downloaded by the script above), the
manifests, the fragment tables and byte statistics, the pairs with their six similarity components and the Govdocs1
external test set. Its README describes every file.

## 2. Training

```
python scripts/32_benchmark_step.py                                     # optional: speed and memory check
python scripts/31_queue.py components main ablations loto robustness   # resumable
```

The queue trains, in this order: one model per similarity component, whose validation R2 values give the
learnability-aware weights (`runs/learnability_weights.json`); the reported models (heuristic weights with seeds 42, 43
and 44; uniform and learnability-aware weights with seed 42); the ablations; a leave-one-type-out model trained without
HEIC; and further seeds (uniform and learnability-aware, seeds 43 and 44) used as a robustness check. Each run is
written to `runs/<run>/` and resumes if it is interrupted. The network is defined in `scripts/model.py` and the
training procedure in `scripts/30_train.py`.

## 3. Evaluation

The evaluation protocol was frozen before any test evaluation. `configs/frozen_protocol.json` defines the models,
baselines, metrics and analyses and records the SHA-256 values of nine scripts. The evaluation scripts stop if one of
these scripts has changed and the change is not logged in `configs/protocol_amendments.log` (see Notes). The protocol is
already frozen, so `40_freeze_protocol.py` is not run again.

```
python scripts/41_evaluate_local.py                     # test split: regression, provenance, shortlist,
                                                        # reconstruction, leave-one-type-out, cost
python scripts/51_govdocs_eval.py                       # Govdocs1 external test set
python scripts/60_bootstrap.py local --ari --workers 3  # bootstrap intervals; each ARI worker needs about 3 GB
python scripts/60_bootstrap.py recon --workers 3
python scripts/60_bootstrap.py govdocs --workers 3
python scripts/61_ablation_paired.py                    # paired ablation comparisons (exploratory; see Notes)
python scripts/70_tables.py                             # results/tables/
python scripts/71_figures.py                            # results/figures/
python scripts/72_submission_figures.py <folder>        # Figures 1-6 at 600 dpi
```

On Windows, `scripts\finish_all.cmd` runs the training queue, the local evaluation (regression, provenance,
reconstruction, leave-one-type-out), the local and reconstruction bootstraps, the paired ablation comparison, the
tables and the figures in one go. The shortlist analysis, the cost measurement and the Govdocs1 steps are run
separately.

## 4. Rebuilding the data set from the sources (optional)

```
python scripts/fetch_archives.py napierone zenodo_heic govdocs1
python scripts/12_verify_napierone_archives.py
python scripts/10_enumerate_commons.py
python scripts/10b_extend_commons_probes.py webm 200
python scripts/11_collect_zenodo.py
python scripts/13_select_and_verify.py
python scripts/15_check_collection.py
python scripts/14_families_and_split.py
python scripts/20_build_fragments.py
python scripts/21_build_pairs.py
python scripts/50_govdocs_build.py
```

Wikimedia Commons is a live repository, so a new enumeration can list newer uploads; exact reconstruction therefore
starts from the frozen manifests of the data set. With the 500 files in place, scripts 15, 14, 20 and 21 reproduce the
data set's collection check, families, split, fragments, byte statistics and pairs exactly (checked on Windows 11
before release).

## Notes

* **Classic zlib.** The compression component and the compression baseline use classic zlib 1.3.1, whose output
  differs from that of zlib-ng. If `python -c "import zlib; print(zlib.ZLIB_RUNTIME_VERSION)"` prints a version that
  contains `ng` (as CPython 3.14 on Windows and some Linux distributions do), `scripts/components.py` runs a separate
  interpreter with classic zlib, set in `CLASSIC_ZLIB_PYTHON`. Point it at a local CPython 3.12 with classic zlib
  1.3.x. Because `components.py` is a frozen script, log the change after editing it:
  `python scripts/40_freeze_protocol.py amend scripts/components.py "classic-zlib interpreter path for this machine"`.
  The data set already contains the byte statistics and pair components computed with classic zlib.
* **Byte-exact files.** `.gitattributes` keeps every file exactly as committed, because the frozen-script check
  compares SHA-256 values. Do not reformat the frozen scripts; log any deliberate change as an amendment, as above.
* **Linux and macOS.** The manifests record stored paths with Windows separators (`accepted_files\png\PNG-001.png`).
  The tools accept them on any system. For scripts 14, 15 and 20 and the reconstruction step of
  `41_evaluate_local.py`, run `python tools/posix_stored_paths.py` once the 500 files are in place: it creates links
  named like the recorded paths, so the scripts run unchanged. The pipeline was run on Windows 11, and this helper
  could not be tested on Linux or macOS before release.
* **Model sets.** The reported models are heuristic seeds 42, 43 and 44, uniform seed 42 and learnability-aware seed
  42. Uniform and learnability-aware seeds 43 and 44 are an additional robustness check.
* **Exploratory analysis.** `61_ablation_paired.py` was added after the protocol was frozen; its results are
  exploratory.
* **Comments that mention an earlier analysis.** Some comments call the data set the "replacement collection" and
  refer to "archived" material: the collection was assembled to replace an earlier, undocumented file collection of
  the same study. The scripts that audited that earlier material are not included, and `ARCHIVE_ROOT` in
  `scripts/paths.py` is a placeholder. In this copy, two docstring lines of `scripts/model.py` that named a local folder
  were shortened; the change is logged in `configs/protocol_amendments.log`.
* **Results.** `results/tables/` holds the result tables of the study, and its README lists the source of each table.
  Trained models and raw predictions are not included.

## Licence

MIT licence (`LICENSE`). The data set has its own licences (see the `LICENSE.md` of the data set).

## Citation

The article reference will be added when it is published. Please also cite the data set and its sources, which are
listed in the data set README.
