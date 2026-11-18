"""PNG validator and repair protocol for the controlled gap-repair experiment (Sections 3.6 and 4.7).

The validator inserts up to ten candidates into the gap in rank order and checks each proposed file
for valid chunk structure, correct CRC checksums and successful decoding.  The first candidate that
passes is accepted; if none passes, the trial is an abstention.  An accepted repair is exact when the
SHA-256 digest of the repaired file equals that of the original; the digest is used only to score
the outcome after the validator's decision.
"""
import hashlib
import io
import struct
import zlib

from .config import FRAGMENT_LENGTH, PNG_MAX_CHECKS

PNG_SIG = b'\x89PNG\r\n\x1a\n'
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def png_chunks(data):
    if not data.startswith(PNG_SIG):
        raise ValueError('no PNG signature')
    off, out, n = 8, [], len(data)
    while off < n:
        if off + 12 > n:
            raise ValueError('truncated chunk')
        length, typ = struct.unpack('>I4s', data[off:off + 8])
        end = off + 12 + length
        if end > n:
            raise ValueError('chunk overruns file')
        crc = struct.unpack('>I', data[end - 4:end])[0]
        out.append({'type': typ.decode('latin-1'), 'offset': off, 'length': length, 'data_start': off + 8,
                    'data_end': off + 8 + length,
                    'crc_ok': (zlib.crc32(data[off + 4:off + 8 + length]) & 0xffffffff) == crc})
        off = end
        if typ == b'IEND':
            break
    return out, off


def _expected_raw(width, height, bitdepth, ctype, interlace):
    bits = _CHANNELS[ctype] * bitdepth
    if not interlace:
        return height * (1 + (width * bits + 7) // 8)
    total = 0
    for x0, y0, dx, dy in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]:
        pw = (width - x0 + dx - 1) // dx if width > x0 else 0
        ph = (height - y0 + dy - 1) // dy if height > y0 else 0
        if pw and ph:
            total += ph * (1 + (pw * bits + 7) // 8)
    return total


def validate_png(data, full_decode=True):
    """Raise ValueError unless `data` is a structurally valid, fully decodable PNG."""
    chunks, end = png_chunks(data)
    if not chunks or chunks[0]['type'] != 'IHDR' or chunks[0]['length'] != 13:
        raise ValueError('IHDR is not the first chunk')
    if chunks[-1]['type'] != 'IEND' or end != len(data):
        raise ValueError('missing IEND or trailing bytes')
    if any(not c['crc_ok'] for c in chunks):
        raise ValueError('CRC mismatch')
    idat = [c for c in chunks if c['type'] == 'IDAT']
    if not idat:
        raise ValueError('no IDAT chunk')
    first, last = chunks.index(idat[0]), chunks.index(idat[-1])
    if any(c['type'] != 'IDAT' for c in chunks[first:last + 1]):
        raise ValueError('IDAT chunks are not consecutive')
    s = chunks[0]['data_start']
    width, height, bitdepth, ctype, comp, filt, inter = struct.unpack('>IIBBBBB', data[s:s + 13])
    if ctype not in _CHANNELS or comp != 0 or filt != 0 or inter not in (0, 1):
        raise ValueError('unsupported IHDR fields')
    expected = _expected_raw(width, height, bitdepth, ctype, inter)
    d, got = zlib.decompressobj(), 0
    for c in idat:
        got += len(d.decompress(data[c['data_start']:c['data_end']]))
        if got > expected:
            raise ValueError('decoded data longer than IHDR implies')
    got += len(d.flush())
    if not d.eof or got != expected:
        raise ValueError('image data does not decode to the size given by IHDR')
    if not inter:
        raw = zlib.decompress(b''.join(data[c['data_start']:c['data_end']] for c in idat))
        if max(raw[::expected // height]) > 4:
            raise ValueError('invalid scanline filter type')
    if full_decode:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 400_000_000
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    return True


def passes(png_bytes):
    try:
        return validate_png(png_bytes, full_decode=True)
    except Exception:  # noqa: BLE001 - any parser or decoder failure rejects the candidate
        return False


def repair_trial(damaged, gap_offset, candidates_in_rank_order, original_sha256, max_checks=PNG_MAX_CHECKS,
                 true_index=None):
    """Run the validator on the first `max_checks` candidates (each a 512-byte bytes object).
    Returns dict(outcome in {'exact', 'wrong', 'abstained'}, accepted_rank, top10_hit)."""
    o = int(gap_offset)
    result = {'outcome': 'abstained', 'accepted_rank': None,
              'top10_hit': None if true_index is None else bool(true_index < max_checks)}
    for r, cand in enumerate(candidates_in_rank_order[:max_checks], 1):
        trial = damaged[:o] + bytes(cand) + damaged[o + FRAGMENT_LENGTH:]
        if passes(trial):
            exact = hashlib.sha256(trial).hexdigest() == original_sha256
            result.update({'outcome': 'exact' if exact else 'wrong', 'accepted_rank': r})
            break
    return result
