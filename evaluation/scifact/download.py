import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent / "data" / "scifact"
BEIR_SCIFACT_ZIP = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)


def download_corpus_and_queries() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")

    corpus.to_json(DATA_DIR / "corpus.jsonl")
    queries.to_json(DATA_DIR / "queries.jsonl")
    print(f"corpus -> {DATA_DIR / 'corpus.jsonl'} ({len(corpus)} rows)")
    print(f"queries -> {DATA_DIR / 'queries.jsonl'} ({len(queries)} rows)")


def download_qrels() -> None:
    """qrels is not on HF BeIR/scifact; take it from the official BEIR zip."""
    qrels_dir = DATA_DIR / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR.parent / "scifact.zip"
    urlretrieve(BEIR_SCIFACT_ZIP, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for name in ("scifact/qrels/train.tsv", "scifact/qrels/test.tsv"):
            target = qrels_dir / Path(name).name
            with zf.open(name) as src, target.open("wb") as dst:
                dst.write(src.read())
            print(f"qrels -> {target}")

    zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_corpus_and_queries()
    download_qrels()
