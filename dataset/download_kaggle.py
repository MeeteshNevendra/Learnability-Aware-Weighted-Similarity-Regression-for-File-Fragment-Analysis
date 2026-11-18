"""Download and extract the public Kaggle release.

Usage:
    python -I dataset/download_kaggle.py --out <folder>

Uses Kaggle's public download endpoint for
https://www.kaggle.com/datasets/meetesh/continuous-fragment-similarity-dataset (about 490 MB).
Alternatively, with the Kaggle command-line tool:
    kaggle datasets download -d meetesh/continuous-fragment-similarity-dataset --unzip -p <folder>
"""
import argparse
import os
import urllib.request
import zipfile

URL = 'https://www.kaggle.com/api/v1/datasets/download/meetesh/continuous-fragment-similarity-dataset'
EXPECTED = {'train.npz', 'val.npz', 'test.npz', 'train_meta.csv', 'val_meta.csv', 'test_meta.csv'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    archive = os.path.join(a.out, 'continuous-fragment-similarity-dataset.zip')
    if not os.path.exists(archive):
        print('downloading', URL)
        req = urllib.request.Request(URL, headers={'User-Agent': 'lawsr-download/1.0'})
        with urllib.request.urlopen(req, timeout=300) as r, open(archive + '.part', 'wb') as fh:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(archive + '.part', archive)
    with zipfile.ZipFile(archive) as z:
        names = {i.filename for i in z.infolist() if not i.is_dir()}
        if not EXPECTED <= names:
            raise SystemExit(f'unexpected archive contents: {sorted(names)}')
        for name in sorted(EXPECTED):            # plain file names only; nothing outside the output folder
            with z.open(name) as src, open(os.path.join(a.out, name), 'wb') as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
    print('extracted', sorted(EXPECTED), 'into', a.out)


if __name__ == '__main__':
    main()
