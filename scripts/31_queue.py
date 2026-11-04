"""Resumable training queue. usage: python scripts/31_queue.py components main ablations loto robustness

Phases (run in the order given on the command line; finished runs are skipped, interrupted runs resume):
  components  one model per component target (seed 42): comp_hist, comp_ent, comp_ngram, comp_comp, comp_pos, comp_type
              -> runs/learnability_weights.json (validation R2 at the lowest-validation-loss epoch only)
  main        reported seed settings: heuristic_s42/s43/s44, uniform_s42, learnability_s42
  ablations   no_hist ... no_type (remaining heuristic weights renormalised) and the archived subset targets
              hist_ent (0.75, 0.25), hist_ngram (0.40, 0.60), hist_ent_comp (0.45, 0.20, 0.35), pos_type (0.60, 0.40);
              single-component targets are the component runs
  loto        loto_heic_s42: heuristic weights, every pair containing a HEIC fragment removed from train and val
  robustness  additional seeds not in the manuscript: uniform_s43/s44, learnability_s43/s44
Learnability rule (Eq. 4): w_k = max(R2_val,k, 0) / sum_m max(R2_val,m, 0). If every component has R2_val <= 0 the
learnability-aware target is undefined; its runs are skipped and recorded as undefined (no fallback weights).
"""
import json
import os
import subprocess
import sys

from paths import LOGS, RUNS, ROOT

PY = sys.executable
TRAIN = os.path.join(ROOT, "scripts", "30_train.py")
HEUR = {"s_hist": 0.25, "s_ent": 0.10, "s_ngram": 0.30, "s_comp": 0.10, "s_pos": 0.15, "s_type": 0.10}
UNIFORM = {k: 1.0 / 6 for k in HEUR}
COMP_RUNS = {"comp_hist": "s_hist", "comp_ent": "s_ent", "comp_ngram": "s_ngram", "comp_comp": "s_comp",
             "comp_pos": "s_pos", "comp_type": "s_type"}
LW_PATH = os.path.join(RUNS, "learnability_weights.json")


def drop(c):
    return {k: v for k, v in HEUR.items() if k != c}


def learnability():
    if os.path.exists(LW_PATH):
        return json.load(open(LW_PATH))
    r2 = {}
    for run, comp in COMP_RUNS.items():
        s = json.load(open(os.path.join(RUNS, run, "summary_val.json")))
        r2[comp] = s["best_val_r2"]
    pos = {k: max(v, 0.0) for k, v in r2.items()}
    tot = sum(pos.values())
    out = {"rule": "w_k = max(R2_val_k, 0) / sum_m max(R2_val_m, 0); validation data only",
           "val_r2": r2, "clipped": pos, "undefined": tot <= 0,
           "weights": {k: v / tot for k, v in pos.items() if v > 0} if tot > 0 else None,
           "sources": {comp: f"runs/{run}/summary_val.json" for run, comp in COMP_RUNS.items()}}
    json.dump(out, open(LW_PATH, "w"), indent=1)
    return out


def phase_runs(phase):
    if phase == "components":
        return [(n, {c: 1.0}, 42, None) for n, c in COMP_RUNS.items()]
    if phase == "main":
        lw = learnability()
        runs = [("heuristic_s42", HEUR, 42, None), ("uniform_s42", UNIFORM, 42, None)]
        runs += [] if lw["undefined"] else [("learnability_s42", lw["weights"], 42, None)]
        return runs + [("heuristic_s43", HEUR, 43, None), ("heuristic_s44", HEUR, 44, None)]
    if phase == "ablations":
        runs = [(f"no_{c[2:]}", drop(c), 42, None) for c in HEUR]
        return runs + [("hist_ent", {"s_hist": 0.75, "s_ent": 0.25}, 42, None),
                       ("hist_ngram", {"s_hist": 0.40, "s_ngram": 0.60}, 42, None),
                       ("hist_ent_comp", {"s_hist": 0.45, "s_ent": 0.20, "s_comp": 0.35}, 42, None),
                       ("pos_type", {"s_pos": 0.60, "s_type": 0.40}, 42, None)]
    if phase == "loto":
        return [("loto_heic_s42", HEUR, 42, "heic")]
    if phase == "robustness":
        lw = learnability()
        runs = [("uniform_s43", UNIFORM, 43, None), ("uniform_s44", UNIFORM, 44, None)]
        return runs + ([] if lw["undefined"] else [("learnability_s43", lw["weights"], 43, None),
                                                   ("learnability_s44", lw["weights"], 44, None)])
    raise ValueError(phase)


def main():
    for phase in sys.argv[1:]:
        for name, w, seed, excl in phase_runs(phase):
            if os.path.exists(os.path.join(RUNS, name, "summary_val.json")):
                print("skip", name, flush=True)
                continue
            cmd = [PY, TRAIN, "--name", name, "--weights", json.dumps(w), "--seed", str(seed)]
            if excl:
                cmd += ["--exclude-ext", excl]
            print(">>>", " ".join(cmd), flush=True)
            with open(os.path.join(LOGS, f"train_{name}.log"), "a", encoding="utf-8") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            print(f"<<< {name} exit={rc}", flush=True)
            if rc != 0:
                print(f"!!! {name} failed; see logs/train_{name}.log", flush=True)
        if phase == "components":
            print("learnability:", json.dumps(learnability()), flush=True)
    print("QUEUE DONE", " ".join(sys.argv[1:]), flush=True)


if __name__ == "__main__":
    main()
