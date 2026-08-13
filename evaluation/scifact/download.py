"""Download and unzip the BEIR SciFact dataset."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATASET = "scifact"
URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"

DATA_ROOT = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_ROOT / f"{DATASET}.zip"
DATASET_DIR = DATA_ROOT / DATASET


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_DIR.exists() and not force:
        print(f"already present: {DATASET_DIR}")
        return DATASET_DIR

    if force and DATASET_DIR.exists():
        for path in sorted(DATASET_DIR.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        DATASET_DIR.rmdir()

    if not ZIP_PATH.exists() or force:
        print(f"downloading {URL}")
        urlretrieve(URL, ZIP_PATH)
    else:
        print(f"using cached zip: {ZIP_PATH}")

    checksum = _md5(ZIP_PATH)
    if checksum != EXPECTED_MD5:
        raise ValueError(
            f"md5 mismatch for {ZIP_PATH}: got {checksum}, expected {EXPECTED_MD5}"
        )

    print(f"extracting {ZIP_PATH} -> {DATA_ROOT}")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_ROOT)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"expected dataset folder missing: {DATASET_DIR}")

    print(f"ready: {DATASET_DIR}")
    return DATASET_DIR


if __name__ == "__main__":
    download()
