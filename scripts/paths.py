"""Shared paths and fixed constants for the replication. Nothing here depends on test results."""
import os
import sys

# Windows consoles default to cp1252; file names from public sources contain arbitrary Unicode.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
MANIFESTS = os.path.join(DATA, "manifests")
ACCEPTED = os.path.join(DATA, "accepted_files")      # accepted source files, renamed by collection id
FRAGMENTS = os.path.join(DATA, "fragments")
PAIRS = os.path.join(DATA, "pairs")
GOVDOCS = os.path.join(DATA, "govdocs1")
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
CONFIGS = os.path.join(ROOT, "configs")
LOGS = os.path.join(ROOT, "logs")
AUDIT = os.path.join(ROOT, "audit")

# Archived (original) material - read only
ARCHIVE_ROOT = r"ARCHIVE_NOT_INCLUDED"  # material of an earlier analysis; not needed here

# Contact details are added only if the person running the scripts sets FSR_CONTACT themselves.
USER_AGENT = "FragmentSimilarityReproduction/1.0 (academic replication" + (
    f"; contact {os.environ['FSR_CONTACT']})" if os.environ.get("FSR_CONTACT") else ")")

for _d in (DATA, RAW, MANIFESTS, ACCEPTED, FRAGMENTS, PAIRS, GOVDOCS, RUNS, RESULTS, CONFIGS, LOGS, AUDIT):
    os.makedirs(_d, exist_ok=True)
