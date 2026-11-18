# Learnability-Aware Weighted Similarity Regression (LAWSR)

Code for the manuscript *Learnability-Aware Weighted Similarity Regression for File Fragment Analysis*.

LAWSR scores a pair of file fragments with a graded provenance score. The score is 0 for blocks from different files. For blocks from the same file it rises as their relative offset separation falls.
- **Inputs:** four byte-level similarities: histogram cosine, entropy similarity, byte 2-gram Jaccard and zlib-based compression similarity.
- **Scaling:** each input is scaled by its learnability, meaning its individual predictive value on validation files.
- **Model:** a compact 64-32 neural regressor combines the scaled inputs.

The repository contains:
- the method;
- every baseline and control described in the manuscript;
- the evaluation protocol: pair metrics, exact-successor retrieval, Govdocs1 transfer and stress conditions, type-label corruption, the controlled PNG gap repair and the component ablation.

The experiments run on the public dataset
[continuous-fragment-similarity-dataset](https://www.kaggle.com/datasets/meetesh/continuous-fragment-similarity-dataset) (Kaggle) and on Govdocs1 archive 001.

## Repository layout

```
lawsr/                  the method (importable package)
  config.py             every protocol constant, with the manuscript section that states it
  similarity.py         Eqs. (7)-(11) byte similarities; Eqs. (15)-(16) fixed fusions; fast pairwise scoring
  reference.py          Eq. (23) provenance reference; Eqs. (12)-(13) type and positional cues
  weighting.py          Eq. (17) learnability index; Eq. (18) softmax weights; ablation renormalization
  model.py              Eqs. (19)-(21) 64-32 network (2,433 parameters with four inputs)
  training.py           Eq. (22) and Algorithm 1: Adam, mini-batches, early stopping, checkpoint selection
  metrics.py            Eq. (24) and AUC / AP / Pearson / Spearman / within-file Spearman / grouped errors
  retrieval.py          Eq. (25) exact-successor ranks with the fixed tie order, Top-k, MRR, P@10
  sampling.py           pair sampling in four modes (Govdocs1 external pairs)
  png_validator.py      PNG validator and the ten-candidate repair protocol
  data.py               loaders for the prepared dataset folder
dataset/
  download_kaggle.py    downloads and extracts the Kaggle release
  prepare_kaggle.py     converts the release into the folder layout read by experiments/
  build_govdocs1.py     Govdocs1 archive-001 external pairs, stress conditions and retrieval pools
experiments/            step0 ... step8 and run_all.py (one script per stage of the protocol)
tests/                  checks against the worked numbers of the manuscript and of the conversion
```

## Installation

```bash
python -m pip install -r requirements.txt      # Python >= 3.10; a CPU build of PyTorch is sufficient
python -I tests/run_tests.py                    # 12 checks
```

## Preparing the data

```bash
# 1. the Kaggle release (train/val/test .npz with fragment bytes, and *_meta.csv)
python -I dataset/download_kaggle.py --out data/kaggle

# 2. convert it (writes data/lawsr/04_pairs, 05_retrieval, 06_type_cue, ... and dataset_info.json)
python -I dataset/prepare_kaggle.py --kaggle-dir data/kaggle --out data/lawsr

# 3. Govdocs1 archive 001 (Table 8): download
#    https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/001.zip,
#    extract its members (without folders) into data/govdocs1_001, then
python -I dataset/build_govdocs1.py data/lawsr data/govdocs1_001
```

What `prepare_kaggle.py` does with the release:

| Item | Treatment |
|---|---|
| Pairs | the released 300,000 / 80,000 / 80,000 pairs, their fragment order (first fragment = first argument) and sampling modes are kept as they are |
| Source lengths | recovered for every file from its same-file pairs, `S = abs(o_a - o_b) / (1 - s_pos)` |
| Target | Eq. (23), `y = z (0.5 + 0.5 (1 - abs(p_a - p_b)))` with `p = offset / S`, from file identity and offsets. This equals `0.5 z + 0.5 s_pos`. The release's own `target_similarity` column is kept in the pair CSVs as `target_released`. `--target released` uses it instead. |
| Similarities | the four components of Eqs. (7)-(11) recomputed from the fragment bytes. The released `s_ngram` and `s_comp` columns do not follow Eqs. (9)-(11), so `--features released` keeps them only for comparison. Both versions are written to the pair CSVs. |
| Retrieval records | pool = every retained test fragment; eligible query = its file's fragment at offset + 512 is retained; seed-42 permutation of fragment identifiers as the tie order |
| Type-label corruption | nested 5% / 10% / 20% selections of the test files (rounded, halves up; 3, 7 and 13 of the 67 test files of the release) |
| PNG gap repair (Table 10) | needs the intact test PNG files, which the release does not contain. Pass them with `--png-sources <folder>`, named `<file_id>.png`. The script checks them against the released fragments. Without them, step 7 is skipped. |

`reports/conversion_report.txt` in the output folder records all checks:
- pair counts;
- identical bytes for every occurrence of a fragment;
- file-disjoint partitions;
- source-length consistency;
- the difference between the released and the recomputed similarities.

## Running the experiments

```bash
python -I experiments/run_all.py --data-root data/lawsr --out results
```

Or run step by step. Each script accepts `--data-root`, `--out`, `--seeds`, `--workers` and `--train-workers`.

| Step | Script | Produces | Manuscript |
|---|---|---|---|
| 0 | `step0_prepare.py` | cached pair arrays; check of the stored similarities against `lawsr.similarity` | Section 3.7 |
| 1 | `step1_learnability.py` | single-component regressors (5 seeds); learnability; weights per temperature | Section 3.3, Eq. 17, Table 5 |
| 2 | `step2_temperature.py` | LAWSR fitted at every temperature; selection by validation MSE; frozen w* | Section 3.3, Eq. 18, Algorithm 1 |
| 3 | `step3_models.py` | unweighted, equal, heuristic, matched binary classifier, four ablations, type-assisted model | Sections 3.5, 3.6, 4.8 |
| 4 | `step4_pair_metrics.py` | metrics on the test pairs; error by mode and band; corrupted type labels | Tables 4, 6, 9, 11 |
| 5 | `step5_retrieval.py` | exact-successor retrieval on the test pool | Table 7, Table 11 (Top-10) |
| 6 | `step6_govdocs1.py` | frozen-model transfer and stress conditions (skipped if step 3 of the data preparation was not run) | Table 8 |
| 7 | `step7_png_repair.py` | controlled PNG gap repair (skipped without the intact test PNGs) | Table 10, Figure 4 |
| 8 | `step8_report.py` | `results/RESULTS.md`, `results/tables/*.csv`, `results/figures/figure3.png` | Tables 4-11, Figure 3 |

Every learned method is fitted with seeds 0-4. Tables report the mean and standard deviation over the seeds. `results/` also keeps:
- every checkpoint, with its frozen input scales;
- the validation and test predictions of every fit;
- the training histories;
- the intermediate metrics as JSON.

**Runtime.** Data preparation takes a few minutes, and so does training (steps 1-3), with parallel fits. Most of the time goes to the retrieval steps:
- Step 6 scores about 25 million query-candidate pairs in each of the seven Govdocs1 conditions.
- The compression component is computed in parallel worker processes (`--workers`), so the run time depends mainly on the number of CPU cores.

**Windows.** Each training process loads PyTorch. If many fits run at once, the Windows commit limit can be exhausted (`WinError 1455`); lower `--train-workers` (the default is 12) if this happens.

## Mapping from the manuscript to the code

| Manuscript | Implementation |
|---|---|
| Eq. (7) histogram cosine; Eq. (8) entropy similarity; Eq. (9) byte 2-gram Jaccard; Eqs. (10)-(11) NCD with zlib level 6, query first | `lawsr.similarity.pair_similarity` (direct transcription); `FragmentStore` and `PairFeaturizer` (same values, fast) |
| Eq. (12) type cue; Eq. (13) positional cue; Eq. (23) provenance reference | `lawsr.reference` |
| Eq. (14) component order (hist, ent, ngram, comp) | `lawsr.config.COMPONENTS` |
| Eq. (15) uniform fusion; Eq. (16) weighted linear fusion with the validation-derived weights | `lawsr.similarity.uniform_fusion`, `weighted_fusion` |
| Eq. (17) learnability index; Eq. (18) softmax, grid 0.05-1.00 | `lawsr.weighting`; `experiments/step1_learnability.py`, `step2_temperature.py` |
| Eqs. (19)-(21) network 64-32, ReLU, sigmoid; 2,433 parameters | `lawsr.model.LAWSRNet` |
| Eq. (22), Algorithm 1: Adam (lr 0.001), batches of 1,024, at most 200 epochs, patience 10 on validation MSE, best checkpoint | `lawsr.training.fit_network` |
| Input-scaling controls: unscaled, equal 0.25, heuristic (0.333, 0.133, 0.400, 0.133) | `lawsr.config`; `experiments/step3_models.py` |
| Matched binary classifier: unscaled network, binary cross-entropy on z; probabilities times the mean positive training reference for graded metrics | `step3_models.py`, `step4_pair_metrics.py` |
| Eq. (24) MSE, RMSE, MAE, R2; Pearson; Spearman; AUC and AP on z; within-file Spearman on the same-file mode | `lawsr.metrics` |
| Eq. (25) Top-1, Top-10, MRR; same-file P@10; ties by the seed-42 permutation of fragment identifiers | `lawsr.retrieval`; records built by `dataset/prepare_kaggle.py` |
| Govdocs1 archive 001 and stress conditions, frozen fits, 76-file retrieval pools | `dataset/build_govdocs1.py`; `experiments/step6_govdocs1.py` |
| Type-label corruption, nested selections of the test files | `dataset/prepare_kaggle.py`; `experiments/step4_pair_metrics.py` |
| Controlled PNG repair: ten-candidate validator; SHA-256 used only for scoring | `dataset/prepare_kaggle.py --png-sources`; `lawsr.png_validator`; `experiments/step7_png_repair.py` |
| Content ablation: remaining scales renormalized to sum to one, refitted | `lawsr.weighting.renormalized`; `step3_models.py` |

## Settings not fixed by the manuscript text

| Setting | Choice | Option |
|---|---|---|
| Learnability with five seeds | validation R2 and Spearman averaged over the five single-component fits | `step1 --policy {mean,seed0}` |
| Temperature selection with five seeds | lowest mean validation MSE over the seeds | `step2 --policy {mean,seed0}` |
| Checkpoint criterion of the binary classifier | validation MSE between probability and z (the shared criterion applied to its own target) | `lawsr/training.py` |
| Scale of the fifth (type-cue) input | 1.0, the four content inputs keep w* | `step3 --type-cue-scale` |
| "One fixed checkpoint" in the PNG experiment | seed 0 | `step7 --checkpoint-seed` |
| "Selected single-component score" in Table 8 | the component with the highest validation learnability | `step6_govdocs1.py` |
| Fragment identifiers for the tie order | `<file_id>_<retained index>` from the release | `dataset/prepare_kaggle.py` |
| Replacement extensions for corrupted labels | drawn (seed 42) from the other extensions recorded in the release | `dataset/prepare_kaggle.py` |
| Initialization and shuffling | PyTorch defaults after `torch.manual_seed(seed)`; one shuffling generator per fit | `lawsr/training.py` |

## Citation

Please cite the article once it is published. Citation details will be added here.

## License

MIT (see `LICENSE`). The dataset is distributed separately under its own terms.
