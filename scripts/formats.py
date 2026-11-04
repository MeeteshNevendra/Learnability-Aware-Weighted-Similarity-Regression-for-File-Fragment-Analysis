"""Content verification for the ten extension groups and features used to detect related versions.

verify(group, data) -> (ok, verified_format, details, reason)
  png        PNG signature + full Pillow decode
  jpg/jpeg   JPEG SOI marker + Pillow decode (Pillow format JPEG or MPO); JPG and JPEG are the same format
  heic       ISO-BMFF 'ftyp' with a HEIF image brand + pillow-heif decode of the primary image
  pdf        '%PDF-' header within the first 1 KiB + pypdf parses at least one page
  mp4        ISO-BMFF top-level box walk covering the file, non-HEIF 'ftyp' brand, 'moov' present
  webm       EBML magic, EBML DocType 'webm', Segment element present
  docx/pptx  ZIP integrity (testzip) + OOXML main part and main content type for the group
  csv        no NUL bytes, decodes as UTF-8 (BOM allowed) or Windows-1252, not markup, csv.Sniffer dialect,
             >= 2 columns in the modal row and >= 80% of the first 500 rows with the modal column count
Nothing is executed from the inspected bytes; parsers only read them.
"""
import csv
import io
import re
import struct
import zipfile

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = 300_000_000
HEIF_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"heif"}


def _fail(reason, **details):
    return False, "", details, reason


# ---------------------------------------------------------------- images
def _image(data, expect):
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            w, h = im.size
            if expect == "JPEG":
                im.draft("L", (max(1, w // 8), max(1, h // 8)))
            if w * h > Image.MAX_IMAGE_PIXELS:
                return None, fmt, (w, h), "decode skipped: more than 300 Mpx"
            im.load()
            return im.copy(), fmt, (w, h), ""
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"decode error: {type(e).__name__}: {str(e)[:120]}"


def verify_png(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return _fail("not PNG signature")
    im, fmt, size, err = _image(data, "PNG")
    if fmt != "PNG" or (im is None and not err.startswith("decode skipped")):
        return _fail("PNG decode failed: " + (err or str(fmt)))
    return True, "PNG image", {"width": size[0], "height": size[1], "note": err}, ""


def verify_jpeg(data):
    if data[:3] != b"\xff\xd8\xff":
        return _fail("not JPEG SOI marker")
    im, fmt, size, err = _image(data, "JPEG")
    if fmt not in ("JPEG", "MPO") or im is None:
        return _fail("JPEG decode failed: " + (err or str(fmt)))
    return True, f"JPEG image ({fmt})", {"width": size[0], "height": size[1]}, ""


def verify_heic(data):
    if data[4:8] != b"ftyp":
        return _fail("no ISO-BMFF ftyp box at offset 4")
    size = struct.unpack(">I", data[:4])[0]
    brands = {data[8:12]} | {data[i:i + 4] for i in range(16, min(size, 64), 4)}
    if not brands & HEIF_BRANDS:
        return _fail(f"ftyp brands {sorted(b.decode('latin-1') for b in brands)} are not HEIF image brands")
    try:
        hf = pillow_heif.open_heif(io.BytesIO(data))
        im = hf.to_pillow()
        w, h = im.size
    except Exception as e:  # noqa: BLE001
        return _fail(f"HEIF decode failed: {type(e).__name__}: {str(e)[:120]}")
    return True, f"HEIF image (major brand {data[8:12].decode('latin-1')})", {"width": w, "height": h,
                                                                            "images": len(hf)}, ""


# ---------------------------------------------------------------- documents
def verify_pdf(data):
    head = data[:1024].find(b"%PDF-")
    if head < 0:
        return _fail("no %PDF- header in first 1 KiB")
    ver = data[head + 5:head + 8].decode("latin-1", "replace")
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data), strict=False)
        enc = bool(r.is_encrypted)
        if enc:
            try:
                r.decrypt("")
            except Exception:  # noqa: BLE001
                pass
        pages = len(r.pages)
    except Exception as e:  # noqa: BLE001
        return _fail(f"pypdf failed: {type(e).__name__}: {str(e)[:120]}")
    if pages < 1:
        return _fail("PDF has no pages")
    return True, f"PDF {ver}", {"pages": pages, "encrypted": enc, "eof_marker_in_last_2KiB": b"%%EOF" in data[-2048:]}, ""


def _ooxml(data, main_part, content_types):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        bad = z.testzip()
    except Exception as e:  # noqa: BLE001
        return None, f"ZIP error: {type(e).__name__}: {str(e)[:100]}"
    if bad is not None:
        return None, f"ZIP CRC error in {bad}"
    names = set(z.namelist())
    if "[Content_Types].xml" not in names or main_part not in names:
        return None, f"missing OOXML parts ({main_part})"
    ct = z.read("[Content_Types].xml").decode("utf-8", "replace")
    found = [c for c in content_types if c in ct]
    if not found:
        return None, "main content type not found: " + "|".join(re.findall(r'ContentType="([^"]*main\+xml)"', ct))[:150]
    return (z, found[0]), ""


def verify_docx(data):
    if data[:4] != b"PK\x03\x04":
        return _fail("not a ZIP container")
    res, err = _ooxml(data, "word/document.xml", ["wordprocessingml.document.main+xml"])
    if res is None:
        return _fail(err)
    return True, "OOXML WordprocessingML document", {"parts": len(res[0].namelist())}, ""


def verify_pptx(data):
    if data[:4] != b"PK\x03\x04":
        return _fail("not a ZIP container")
    res, err = _ooxml(data, "ppt/presentation.xml", ["presentationml.presentation.main+xml"])
    if res is None:
        return _fail(err)
    slides = sum(1 for n in res[0].namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n))
    return True, "OOXML PresentationML presentation", {"parts": len(res[0].namelist()), "slides": slides}, ""


def verify_csv(data):
    if b"\x00" in data:
        return _fail("NUL bytes (binary content)")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return _fail("not decodable as UTF-8 or Windows-1252")
    if text.lstrip()[:1] in ("<", "{", "["):
        return _fail("markup/JSON content, not delimited text")
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    rows = []
    for i, row in enumerate(csv.reader(io.StringIO(text[:2_000_000]), delimiter=delim)):
        if i >= 500:
            break
        if row:
            rows.append(len(row))
    if len(rows) < 2:
        return _fail("fewer than 2 non-empty rows")
    modal = max(set(rows), key=rows.count)
    share = rows.count(modal) / len(rows)
    if modal < 2 or share < 0.8:
        return _fail(f"inconsistent table structure (modal columns {modal}, share {share:.2f})")
    return True, "delimited text (CSV)", {"encoding": enc, "delimiter": delim, "modal_columns": modal,
                                          "rows_checked": len(rows), "modal_share": round(share, 3)}, ""


# ---------------------------------------------------------------- video containers
def iso_boxes(data):
    """Top-level ISO-BMFF boxes [(type, offset, size)]; raises ValueError if the chain is broken."""
    out, off, n = [], 0, len(data)
    while off < n:
        if n - off < 8:
            raise ValueError(f"{n - off} trailing bytes")
        size, typ = struct.unpack(">I4s", data[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = n - off
        if size < hdr or off + size > n:
            raise ValueError(f"box {typ!r} at {off} overruns the file")
        out.append((typ.decode("latin-1"), off, size))
        off += size
    return out


def mp4_duration_ms(data, boxes):
    for typ, off, size in boxes:
        if typ != "moov":
            continue
        moov = data[off + 8:off + size]
        i = moov.find(b"mvhd")
        if i < 4:
            return None
        ver = moov[i + 4]
        if ver == 1:
            ts, dur = struct.unpack(">IQ", moov[i + 24:i + 36])
        else:
            ts, dur = struct.unpack(">II", moov[i + 16:i + 24])
        return round(1000 * dur / ts) if ts else None
    return None


def verify_mp4(data):
    if data[4:8] != b"ftyp":
        return _fail("no ftyp box at offset 4")
    try:
        boxes = iso_boxes(data)
    except ValueError as e:
        return _fail(f"broken ISO-BMFF box chain: {e}")
    brand = data[8:12]
    if brand in HEIF_BRANDS:
        return _fail(f"HEIF image brand {brand!r}, not a video file")
    types = [b[0] for b in boxes]
    if "moov" not in types:
        return _fail("no moov box")
    return True, f"ISO-BMFF video (major brand {brand.decode('latin-1')})", {
        "top_level_boxes": types[:12], "fragmented": "moof" in types, "duration_ms": mp4_duration_ms(data, boxes)}, ""


def _ebml_vint(b, i, keep_marker=False):
    first = b[i]
    length = 1
    mask = 0x80
    while length <= 8 and not first & mask:
        mask >>= 1
        length += 1
    if length > 8:
        raise ValueError("bad EBML vint")
    val = first if keep_marker else first & (mask - 1)
    for k in range(1, length):
        val = (val << 8) | b[i + k]
    return val, length


def webm_info(data):
    """DocType from the EBML header, presence of the Segment element, and (best effort) Segment/Info duration in ms.
    The Info element is found by walking the Segment's child elements (not by byte search, which can hit the copy
    of the Info ID stored in the SeekHead index)."""
    i = 0
    eid, l = _ebml_vint(data, i, keep_marker=True)
    if eid != 0x1A45DFA3:
        raise ValueError("no EBML header")
    i += l
    size, l = _ebml_vint(data, i)
    i += l
    end = i + size
    doctype = None
    while i < end:
        cid, l = _ebml_vint(data, i, keep_marker=True)
        i += l
        csize, l = _ebml_vint(data, i)
        i += l
        if cid == 0x4282:
            doctype = data[i:i + csize].decode("latin-1").rstrip("\x00")   # EBML strings may be zero-padded
        i += csize
    seg, l = _ebml_vint(data, i, keep_marker=True)
    if seg != 0x18538067:
        raise ValueError("no Segment after EBML header")
    duration = None
    try:
        k = i + l
        ssize, l2 = _ebml_vint(data, k)
        k += l2
        seg_end = len(data) if ssize == (1 << (7 * l2)) - 1 else min(len(data), k + ssize)
        while k < seg_end:
            cid, l3 = _ebml_vint(data, k, keep_marker=True)
            k += l3
            csize, l4 = _ebml_vint(data, k)
            k += l4
            if cid == 0x1549A966:                       # Info
                stop, scale, raw = k + csize, 1_000_000, None
                while k < stop:
                    ccid, a = _ebml_vint(data, k, keep_marker=True)
                    k += a
                    ccsize, b = _ebml_vint(data, k)
                    k += b
                    if ccid == 0x2AD7B1:
                        scale = int.from_bytes(data[k:k + ccsize], "big")
                    elif ccid == 0x4489 and ccsize in (4, 8):
                        raw = struct.unpack(">f" if ccsize == 4 else ">d", data[k:k + ccsize])[0]
                    k += ccsize
                if raw is not None:
                    duration = round(raw * scale / 1e6)
                break
            if cid == 0x1F43B675:                       # Cluster reached: no Info before media data
                break
            k += csize
    except Exception:  # noqa: BLE001 - duration is only used to detect related versions
        duration = None
    return doctype, duration


def verify_webm(data):
    if data[:4] != b"\x1a\x45\xdf\xa3":
        return _fail("no EBML magic")
    try:
        doctype, dur = webm_info(data)
    except Exception as e:  # noqa: BLE001
        return _fail(f"EBML parse error: {e}")
    if doctype != "webm":
        return _fail(f"EBML DocType {doctype!r} is not webm")
    return True, "WebM (EBML DocType webm)", {"duration_ms": dur}, ""


VERIFY = {"png": verify_png, "jpg": verify_jpeg, "jpeg": verify_jpeg, "heic": verify_heic, "pdf": verify_pdf,
          "mp4": verify_mp4, "webm": verify_webm, "docx": verify_docx, "pptx": verify_pptx, "csv": verify_csv}


def verify(group, data):
    try:
        return VERIFY[group](data)
    except Exception as e:  # noqa: BLE001
        return _fail(f"verifier exception: {type(e).__name__}: {str(e)[:120]}")


# ---------------------------------------------------------------- related-version features
def dhash(data):
    """64-bit difference hash of the decoded image (grey, 9x8), or None if not decodable."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format == "JPEG":
                im.draft("L", (64, 64))
            if im.size[0] * im.size[1] > Image.MAX_IMAGE_PIXELS:
                return None
            g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            px = list(g.getdata())
    except Exception:  # noqa: BLE001
        return None
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def _dhash_image(im):
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def video_frame_hashes(data, positions=(0.25, 0.5, 0.75)):
    """dHash of the first decodable video frame at or after each relative position (PyAV/FFmpeg decoding of the
    in-memory bytes). None if the file has no decodable video stream."""
    import av
    try:
        with av.open(io.BytesIO(data)) as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is None:
                return None
            dur = c.duration / 1e6 if c.duration else (float(vs.duration * vs.time_base) if vs.duration else None)
            out = []
            for p in positions:
                if dur:
                    c.seek(int(p * dur * 1e6), backward=True, any_frame=False)
                frame = next(c.decode(vs), None)
                if frame is None:
                    return None
                out.append(_dhash_image(frame.to_image()))
            return out
    except Exception:  # noqa: BLE001
        return None


def document_text(group, data, max_chars=200_000):
    try:
        if group in ("docx", "pptx"):
            z = zipfile.ZipFile(io.BytesIO(data))
            parts = ["word/document.xml"] if group == "docx" else sorted(
                n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n))
            xml = " ".join(z.read(p).decode("utf-8", "replace") for p in parts if p in z.namelist())
            return " ".join(re.findall(r"<(?:w|a):t[^>]*>([^<]*)</(?:w|a):t>", xml))[:max_chars]
        if group == "pdf":
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(data), strict=False)
            return " ".join((p.extract_text() or "") for p in r.pages[:5])[:max_chars]
        if group == "csv":
            return data[:max_chars].decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""
    return ""


def shingles(text, k=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}
