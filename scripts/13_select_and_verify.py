"""Collection protocol v1: verify candidates in a fixed order and accept 50 files per extension group.

Group order: PNG, JPG, HEIC, PDF, MP4, DOCX, PPTX, JPEG, WEBM, CSV (Table 2 order).
Candidate order per group (fixed, documented in the README):
  PNG, JPG, JPEG, WEBM  Wikimedia Commons candidates in probe order (scripts/10_enumerate_commons.py)
  HEIC                  Zenodo candidates in the order of scripts/11_collect_zenodo.py
  PDF, DOCX, PPTX       NapierOne "<TYPE>-tiny.zip" members, sorted by SHA-256("napierone-order-v1|zip|member")
  MP4                   NapierOne "MP4-FROM-WEB-tiny.zip" (files as published on websites, via CommonCrawl), same order
  CSV                   NapierOne "CSV-tiny.zip", then members of "CSV-small.zip" whose names were not already examined
Rules, applied to each candidate in order until the quota is met (first failing rule is the logged reason):
  R1 retrieval: Commons download must succeed and match the enumerated byte size and SHA-1
  R2 size: 1,024 <= bytes <= 20 MiB
  R3 format: content verification for the group (scripts/formats.py) - rejects failed downloads, HTML error pages,
     corrupt/truncated files and misleading extensions
  R4 duplicate: SHA-256 equal to an already accepted file anywhere in the collection
Accepted files are copied to data/accepted_files/<group>/<collection_id><ext>. No file is converted, renamed in its
manifest record, generated or reused across groups. A group that cannot reach 50 is reported as a shortfall.
Outputs: data/manifests/source_manifest_unsplit.csv, exclusion_log.csv, family_features.json
"""
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import time
import zipfile

import pandas as pd
import requests

import formats as F
from paths import ACCEPTED, MANIFESTS, RAW, USER_AGENT

GROUPS = ["png", "jpg", "heic", "pdf", "mp4", "docx", "pptx", "jpeg", "webm", "csv"]
QUOTA = 50
MIN_SIZE, MAX_SIZE = 1024, 20 * 1024 * 1024
NAPIER_ZIPS = {"pdf": ["PDF-tiny.zip"], "mp4": ["MP4-FROM-WEB-tiny.zip"], "docx": ["DOCX-tiny.zip"],
               "pptx": ["PPTX-tiny.zip"], "csv": ["CSV-tiny.zip", "CSV-small.zip"]}
NAPIER_URL = {"PDF-tiny.zip": "PDF/PDF-tiny.zip", "MP4-FROM-WEB-tiny.zip": "MP4/MP4-FROM-WEB-tiny.zip",
              "DOCX-tiny.zip": "DOCX/DOCX-tiny.zip", "PPTX-tiny.zip": "PPTX/PPTX-tiny.zip",
              "CSV-tiny.zip": "CSV/CSV-tiny.zip", "CSV-small.zip": "CSV/CSV-small.zip"}
NAPIER_LICENCE = {
    "pdf": "Edinburgh Napier University Licence Agreement; source files from gov.uk domains (Open Government Licence v3.0 per NapierOne documentation)",
    "docx": "Edinburgh Napier University Licence Agreement; source files from gov.uk domains (Open Government Licence v3.0 per NapierOne documentation)",
    "pptx": "Edinburgh Napier University Licence Agreement; source files from gov.uk domains (Open Government Licence v3.0 per NapierOne documentation)",
    "csv": "Edinburgh Napier University Licence Agreement; source files from gov.uk domains (Open Government Licence v3.0 per NapierOne documentation)",
    "mp4": "Edinburgh Napier University Licence Agreement; files gathered from websites via CommonCrawl (CommonCrawl terms of use); underlying copyright not stated",
}
S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def sha(data, algo="sha256"):
    return hashlib.new(algo, data).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


def archive_log():
    p = os.path.join(MANIFESTS, "archive_retrieval_log.csv")
    return {r["local_path"].replace("\\", "/"): r for r in csv.DictReader(open(p, encoding="utf-8"))}


def commons_candidates(group):
    if group == "webm":   # the WEBM probe sequence was extended (10b_extend_commons_probes.py); wait for completion
        log = os.path.join(MANIFESTS, "candidates_commons_extensions.log")
        while not (os.path.exists(log) and "webm:" in open(log, encoding="utf-8").read()):
            time.sleep(30)
    d = pd.read_csv(os.path.join(MANIFESTS, "candidates_commons.csv"), dtype=str, keep_default_na=False)
    d = d[(d.group == group) & (d.status == "candidate")].copy()
    d["probe_index"] = d.probe_index.astype(int)
    for _, r in d.sort_values("probe_index").iterrows():
        yield {"source": "Wikimedia Commons", "kind": "commons", "order": int(r.probe_index), **r.to_dict()}


def zenodo_candidates():
    d = pd.read_csv(os.path.join(MANIFESTS, "candidates_zenodo_heic.csv"), dtype=str, keep_default_na=False)
    for _, r in d.sort_values("candidate_order", key=lambda s: s.astype(int)).iterrows():
        yield {"source": "Zenodo", "kind": "zenodo", "order": int(r.candidate_order), **r.to_dict()}


def napier_candidates(group, examined_names):
    k = 0
    for zname in NAPIER_ZIPS[group]:
        path = os.path.join(RAW, "napierone", "archives", zname)
        z = zipfile.ZipFile(path)
        members = [i.filename for i in z.infolist() if not i.is_dir()
                   and re.match(r"^\d+-", os.path.basename(i.filename))
                   and i.filename.rsplit(".", 1)[-1].lower() == group]
        members.sort(key=lambda m: hashlib.sha256(f"napierone-order-v1|{zname}|{m}".encode()).hexdigest())
        for m in members:
            if os.path.basename(m) in examined_names:
                continue
            examined_names.add(os.path.basename(m))
            yield {"source": "NapierOne", "kind": "napierone", "order": k, "zip": zname, "member": m}
            k += 1


def fetch_bytes(c, alog):
    """Returns (data, provenance dict) or raises ValueError(reason)."""
    if c["kind"] == "commons":
        dest = os.path.join(RAW, "commons", c["group"], f"{int(c['probe_index']):03d}__{safe(c['title'])}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and sha(open(dest, "rb").read(), "sha1") == c["sha1"]:
            data, when, status = open(dest, "rb").read(), json.load(open(dest + ".json"))["retrieved_utc"], 200
        else:
            data, status = None, None
            for attempt in range(5):
                try:
                    r = S.get(c["url"], timeout=120)
                    status = r.status_code
                    if status == 200:
                        data = r.content
                        break
                    if status in (403, 404, 410):
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5 * (attempt + 1))
            if data is None:
                raise ValueError(f"R1 retrieval failed (HTTP {status})")
            when = now()
            open(dest, "wb").write(data)
            json.dump({"url": c["url"], "retrieved_utc": when, "http_status": status}, open(dest + ".json", "w"))
            time.sleep(0.5)
        if len(data) != int(c["size"]):
            raise ValueError(f"R1 size {len(data)} differs from enumerated {c['size']}")
        if sha(data, "sha1") != c["sha1"]:
            raise ValueError("R1 SHA-1 differs from the enumerated file revision")
        lic = c.get("license_short") or ""
        prov = {"original_filename": c["title"], "source_name": "Wikimedia Commons",
                "source_dataset_release": f"Wikimedia Commons file revision uploaded {c['upload_timestamp']} (SHA-1 {c['sha1']})",
                "source_page": c.get("source_page") or "https://commons.wikimedia.org/wiki/File:" + c["title"],
                "direct_download_url": c["url"], "archive_member": "", "retrieval_utc": when,
                "license": lic, "license_url": c.get("license_url", ""),
                "attribution": "; ".join(x for x in (c.get("artist", ""), c.get("credit", "")) if x),
                "reuse_restrictions": c.get("restrictions", ""),
                "redistribution": ("permitted under the stated free licence with attribution" if lic else
                                   "unclear (no licence metadata) - not redistributed"),
                "candidate_detail": f"probe {c['probe_index']} '{c['probe']}'"}
        return data, prov
    if c["kind"] == "zenodo":
        if c["archive"]:
            zpath = os.path.join(os.path.dirname(RAW), *c["archive"].split("/")[1:])
            data = zipfile.ZipFile(zpath).read(c["file_key"])
            rel = c["archive"].replace("data/raw/", "")
            when = alog.get(rel, {}).get("retrieved_utc", c["metadata_retrieved_utc"])
            member = c["file_key"]
        else:
            data = open(os.path.join(os.path.dirname(RAW), c["local_path"]), "rb").read()
            when, member = c["metadata_retrieved_utc"], ""
            if "md5:" + sha(data, "md5") != c["published_md5"]:
                raise ValueError("R1 MD5 differs from the Zenodo record")
        prov = {"original_filename": os.path.basename(c["file_key"]), "source_name": "Zenodo",
                "source_dataset_release": f"{c['record_title']} (Zenodo record {c['record_id']}, DOI {c['doi']}, "
                                          f"published {c['publication_date']}{', version ' + c['record_version'] if c['record_version'] else ''})",
                "source_page": c["source_page"], "direct_download_url": c["direct_download_url"],
                "archive_member": member, "retrieval_utc": when, "license": c["license"],
                "license_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": c["creators"],
                "reuse_restrictions": "", "redistribution": "permitted under CC BY 4.0 with attribution",
                "candidate_detail": f"order {c['candidate_order']}"}
        return data, prov
    zpath = os.path.join(RAW, "napierone", "archives", c["zip"])
    data = zipfile.ZipFile(zpath).read(c["member"])
    rec = alog.get(f"napierone/archives/{c['zip']}", {})
    grp = c["member"].rsplit(".", 1)[-1].lower()
    prov = {"original_filename": os.path.basename(c["member"]), "source_name": "NapierOne",
            "source_dataset_release": f"NapierOne (Davies et al. 2022, FSI:DI 40:301330) subset {c['zip']} "
                                      f"(archive SHA-256 {rec.get('sha256', '?')})",
            "source_page": "https://registry.opendata.aws/napierone/",
            "direct_download_url": "https://s3.eu-north-1.amazonaws.com/napierone.com/NapierOne/Data/" + NAPIER_URL[c["zip"]],
            "archive_member": c["member"], "retrieval_utc": rec.get("retrieved_utc", ""),
            "license": NAPIER_LICENCE[grp], "license_url": "https://github.com/simonrdavies/NapierOne",
            "attribution": "Edinburgh Napier University (NapierOne data set)", "reuse_restrictions":
            "original per-file source URL not recorded by NapierOne",
            "redistribution": "not redistributed here; retrieve the NapierOne archive and extract the member",
            "candidate_detail": f"{c['zip']} order {c['order']}"}
    return data, prov


def family_features(group, data):
    feat = {}
    if group in ("png", "jpg", "jpeg", "heic"):
        feat["dhash"] = F.dhash(data)
    if group in ("pdf", "docx", "pptx", "csv"):
        feat["shingles"] = sorted(F.shingles(F.document_text(group, data)))
    return feat


def main():
    alog = archive_log()
    accepted_sha = {}
    manifest, exclusions, features = [], [], {}
    shortfalls = {}
    for g in GROUPS:
        os.makedirs(os.path.join(ACCEPTED, g), exist_ok=True)
        if g in ("png", "jpg", "jpeg", "webm"):
            cands = commons_candidates(g)
        elif g == "heic":
            cands = zenodo_candidates()
        else:
            cands = napier_candidates(g, set())
        n_acc, n_seen = 0, 0
        for c in cands:
            if n_acc >= QUOTA:
                break
            n_seen += 1
            c.setdefault("group", g)
            row = {"group": g, "examined_rank": n_seen, "source": c["source"],
                   "candidate": c.get("title") or c.get("file_key") or c.get("member"), "decision": "", "reason": "",
                   "bytes": "", "sha256": ""}
            try:
                data, prov = fetch_bytes(c, alog)
            except ValueError as e:
                exclusions.append({**row, "decision": "excluded", "reason": str(e)})
                continue
            row.update({"bytes": len(data), "sha256": sha(data)})
            ext = prov["original_filename"].rsplit(".", 1)[-1].lower() if "." in prov["original_filename"] else ""
            if ext != g:
                exclusions.append({**row, "decision": "excluded", "reason": f"recorded extension '{ext}' is not group '{g}'"})
                continue
            if not MIN_SIZE <= len(data) <= MAX_SIZE:
                exclusions.append({**row, "decision": "excluded", "reason": f"R2 size {len(data)} outside [1024, 20 MiB]"})
                continue
            ok, fmt, details, reason = F.verify(g, data)
            if not ok:
                exclusions.append({**row, "decision": "excluded", "reason": "R3 " + reason})
                continue
            h = sha(data)
            if h in accepted_sha:
                exclusions.append({**row, "decision": "excluded", "reason": f"R4 byte-identical duplicate of {accepted_sha[h]}"})
                continue
            n_acc += 1
            cid = f"{g.upper()}-{n_acc:03d}"
            accepted_sha[h] = cid
            stored = os.path.join(ACCEPTED, g, f"{cid}.{ext}")
            open(stored, "wb").write(data)
            manifest.append({"collection_id": cid, "extension_group": g.upper(), "recorded_extension": ext,
                             "verified_format": fmt, "format_details": json.dumps(details, default=str),
                             "size_bytes": len(data), "sha256": h, "sha1": sha(data, "sha1"), "md5": sha(data, "md5"),
                             **prov, "stored_path": os.path.relpath(stored, os.path.dirname(ACCEPTED)),
                             "acceptance_rank_in_group": n_acc})
            features[cid] = family_features(g, data)
            exclusions.append({**row, "decision": "accepted", "reason": cid})
            print(g, cid, prov["original_filename"][:60], len(data), fmt, flush=True)
        if n_acc < QUOTA:
            shortfalls[g] = {"accepted": n_acc, "required": QUOTA, "candidates_examined": n_seen}
            print("SHORTFALL", g, n_acc, flush=True)
    pd.DataFrame(manifest).to_csv(os.path.join(MANIFESTS, "source_manifest_unsplit.csv"), index=False)
    pd.DataFrame(exclusions).to_csv(os.path.join(MANIFESTS, "exclusion_log.csv"), index=False)
    json.dump(features, open(os.path.join(MANIFESTS, "family_features.json"), "w"))
    json.dump({"shortfalls": shortfalls, "accepted_total": len(manifest), "completed_utc": now()},
              open(os.path.join(MANIFESTS, "selection_summary.json"), "w"), indent=1)
    print("SELECTION COMPLETE", len(manifest), "shortfalls:", shortfalls, flush=True)


if __name__ == "__main__":
    main()
