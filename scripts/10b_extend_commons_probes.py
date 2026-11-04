"""Extend the seeded probe sequence of one Commons group (continuation of 10_enumerate_commons.py).

Reason (recorded before any WEBM candidate was downloaded or verified): only 51 of 110 WEBM probes found a .webm file
within 2,000 names, leaving one spare candidate for a quota of 50. The same generator ("commons-<group>-v1") is
advanced past the existing probes and continued up to the new probe count; earlier candidates keep their order.
usage: python scripts/10b_extend_commons_probes.py webm 200
"""
import csv
import importlib
import os
import random
import sys
import time

from paths import MANIFESTS

E = importlib.import_module("10_enumerate_commons")


def main():
    group, total = sys.argv[1], int(sys.argv[2])
    path = os.path.join(MANIFESTS, "candidates_commons.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fields = list(rows[0].keys())
    done = sorted(int(r["probe_index"]) for r in rows if r["group"] == group)
    rng = random.Random(f"commons-{group}-v1")
    seen = {r["title"] for r in rows if r["group"] == group and r["status"] == "candidate"}
    new = []
    for k in range(total):
        start = rng.choice(E.FIRST) + "".join(rng.choice(E.REST) for _ in range(2))
        if k <= done[-1]:
            continue
        x, page = E.probe(start, group)
        if x is None:
            new.append({"group": group, "probe_index": k, "probe": start, "status": "no_match_within_2000_names"})
            continue
        if x["name"] in seen:
            new.append({"group": group, "probe_index": k, "probe": start, "status": "already_candidate", "title": x["name"]})
            continue
        seen.add(x["name"])
        new.append({"group": group, "probe_index": k, "probe": start, "status": "candidate", "title": x["name"],
                    "url": x["url"].split("?")[0], "size": x["size"], "mime": x["mime"], "sha1": x["sha1"],
                    "upload_timestamp": x["timestamp"], "uploader": x.get("user", ""), "pages_scanned": page + 1})
        print(group, k, start, x["name"], x["size"], flush=True)
        time.sleep(0.2)
    cands = [r for r in new if r["status"] == "candidate"]
    lic = E.licence(["File:" + r["title"] for r in cands])
    for r in cands:
        m = lic.get("File:" + r["title"].replace("_", " "), lic.get("File:" + r["title"], {}))
        r.update({"source_page": m.get("descriptionurl"), "license_short": m.get("LicenseShortName"),
                  "license_url": m.get("LicenseUrl"), "usage_terms": m.get("UsageTerms"),
                  "attribution_required": m.get("AttributionRequired"), "artist": E.strip_html(m.get("Artist")),
                  "credit": E.strip_html(m.get("Credit")), "restrictions": m.get("Restrictions"),
                  "copyrighted": m.get("Copyrighted"), "license_checked_sha1": m.get("ii_sha1")})
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for r in new:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(os.path.join(MANIFESTS, "candidates_commons_extensions.log"), "a", encoding="utf-8") as f:
        f.write(f"{group}: probes {done[-1] + 1}..{total - 1} added ({len(cands)} new candidates); reason: low "
                f"candidate yield before selection (51 of 110 probes matched)\n")
    print("EXTENSION COMPLETE", group, len(cands), flush=True)


if __name__ == "__main__":
    main()
