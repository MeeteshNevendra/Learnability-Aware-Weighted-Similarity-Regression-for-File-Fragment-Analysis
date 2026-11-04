"""Standard-library-only helper: zlib-compressed lengths of fixed-size blocks.

Run with an interpreter linked against classic zlib (here CPython 3.12.10, zlib 1.3.1). CPython 3.14 on Windows links
zlib-ng 2.2.4, whose level-6 output differs from classic zlib for ~8% of archived fragment pairs; classic zlib 1.2.13
and 1.3.1 reproduce every archived s_comp value exactly (audit/zlib_check.json).

usage: python zlib_classic_lengths.py <in.bin> <block_len> <level> <out.bin>   (out: little-endian uint32 per block)
"""
import struct
import sys
import zlib


def main():
    src, block, level, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    data = open(src, "rb").read()
    lens = [len(zlib.compress(data[i:i + block], level)) for i in range(0, len(data), block)]
    with open(dst, "wb") as f:
        f.write(struct.pack(f"<{len(lens)}I", *lens))
    sys.stderr.write(f"zlib {zlib.ZLIB_RUNTIME_VERSION} blocks={len(lens)}\n")


if __name__ == "__main__":
    main()
