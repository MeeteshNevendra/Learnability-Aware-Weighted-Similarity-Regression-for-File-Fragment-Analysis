"""Enumerate Wikimedia Commons candidates for the PNG, JPG, JPEG and WEBM extension groups (metadata only).

Fixed rule (collection protocol v1):
  * For extension group e, a pseudo-random generator seeded with the string "commons-<e>-v1" produces probe strings:
    first character from A-Z0-9, then two characters from a-z.
  * For each probe, list=allimages is queried in name order starting at the probe (aiminsize=1024,
    aimaxsize=20 MiB, 500 names per page, at most 4 pages). The first file whose lower-cased filename extension equals
    e becomes that probe's candidate. Probes without a match within 2,000 names are logged as "no match".
  * Candidate order = probe order. scripts/13_select_and_verify.py downloads candidates in this order and accepts
    files until the group quota is filled; exclusions are logged there.
  * imageinfo/extmetadata supplies the licence, author, upload timestamp and SHA-1 of the file revision.
Output: data/manifests/candidates_commons.csv
"""
import csv
import os
import random
import time

import requests

from paths import MANIFESTS, USER_AGENT

API = "https://commons.wikimedia.org/w/api.php"
GROUPS = {"png": 110, "jpg": 110, "jpeg": 110, "webm": 110}   # probes per group (upper bound; quota is 50)
MAX_SIZE = 20 * 1024 * 1024
FIRST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
REST = "abcdefghijklmnopqrstuvwxyz"
S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def api(params):
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(6):
        try:
            r = S.get(API, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("API failed")


def probe(start, ext):
    params = {"action": "query", "list": "allimages", "aisort": "name", "aifrom": start, "ailimit": 500,
              "aiminsize": 1024, "aimaxsize": MAX_SIZE, "aiprop": "url|size|mime|sha1|timestamp|user"}
    for page in range(4):
        d = api(params)
        for x in d["query"]["allimages"]:
            name = x["name"]
            if "." in name and name.rsplit(".", 1)[1].lower() == ext:
                return x, page
        if "continue" not in d:
            return None, page
        params.update(d["continue"])
        time.sleep(0.2)
    return None, 3


def licence(titles):
    out = {}
    for i in range(0, len(titles), 40):
        d = api({"action": "query", "prop": "imageinfo", "titles": "|".join(titles[i:i + 40]),
                 "iiprop": "url|sha1|size|mime|timestamp|user|extmetadata",
                 "iiextmetadatafilter": "LicenseShortName|LicenseUrl|UsageTerms|AttributionRequired|Artist|Credit|"
                                        "Copyrighted|Restrictions|DateTimeOriginal|ImageDescription"})
        norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
        for p in d["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            em = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in ii.get("extmetadata", {}).items()}
            out[norm.get(p["title"], p["title"])] = {**em, "descriptionurl": ii.get("descriptionurl"),
                                                     "ii_sha1": ii.get("sha1"), "ii_timestamp": ii.get("timestamp")}
        time.sleep(0.3)
    return out


def strip_html(s):
    import re
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())[:300]


def main():
    rows = []
    for ext, n_probes in GROUPS.items():
        rng = random.Random(f"commons-{ext}-v1")
        seen = set()
        for k in range(n_probes):
            start = rng.choice(FIRST) + "".join(rng.choice(REST) for _ in range(2))
            x, page = probe(start, ext)
            if x is None:
                rows.append({"group": ext, "probe_index": k, "probe": start, "status": "no_match_within_2000_names"})
                continue
            if x["name"] in seen:
                rows.append({"group": ext, "probe_index": k, "probe": start, "status": "already_candidate",
                             "title": x["name"]})
                continue
            seen.add(x["name"])
            rows.append({"group": ext, "probe_index": k, "probe": start, "status": "candidate", "title": x["name"],
                         "url": x["url"].split("?")[0], "size": x["size"], "mime": x["mime"], "sha1": x["sha1"],
                         "upload_timestamp": x["timestamp"], "uploader": x.get("user", ""), "pages_scanned": page + 1})
            print(ext, k, start, x["name"], x["size"], flush=True)
            time.sleep(0.2)
    cands = [r for r in rows if r["status"] == "candidate"]
    lic = licence(["File:" + r["title"] for r in cands])
    for r in cands:
        m = lic.get("File:" + r["title"].replace("_", " "), lic.get("File:" + r["title"], {}))
        r.update({"source_page": m.get("descriptionurl"), "license_short": m.get("LicenseShortName"),
                  "license_url": m.get("LicenseUrl"), "usage_terms": m.get("UsageTerms"),
                  "attribution_required": m.get("AttributionRequired"), "artist": strip_html(m.get("Artist")),
                  "credit": strip_html(m.get("Credit")), "restrictions": m.get("Restrictions"),
                  "copyrighted": m.get("Copyrighted"), "license_checked_sha1": m.get("ii_sha1")})
    fields = ["group", "probe_index", "probe", "status", "title", "url", "source_page", "size", "mime", "sha1",
              "license_checked_sha1", "upload_timestamp", "uploader", "pages_scanned", "license_short", "license_url",
              "usage_terms", "attribution_required", "artist", "credit", "restrictions", "copyrighted"]
    with open(os.path.join(MANIFESTS, "candidates_commons.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("ENUMERATION COMPLETE", {e: sum(1 for r in cands if r["group"] == e) for e in GROUPS}, flush=True)


if __name__ == "__main__":
    main()
