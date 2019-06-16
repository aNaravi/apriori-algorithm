import os
import gzip
import shutil
import argparse
import requests

LINKS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.kos.txt.gz",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.pubmed.txt.gz",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/readme.txt",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.enron.txt",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.kos.txt",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nips.txt",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nytimes.txt",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.pubmed.txt"
]

def main(prefix: str, verbose: int = 0) -> None:
    os.makedirs(prefix, exist_ok=True)

    for link in LINKS:
        filename = link.split('/')[-1]
        if verbose: print(f"[INFO] Downloading {filename}")
        with open(prefix + '/' + filename, 'wb') as fp:
            fp.write(requests.get(link, stream=True).content)

    for filename in os.listdir(prefix):
        filename = os.path.join(prefix, filename)
        basename, ext = os.path.splitext(filename)
        if ext == ".gz":
            if verbose: print(f"[INFO] Decompressing {basename}")
            with gzip.open(filename, 'rb') as rp, open(basename, 'wb') as wp:
                shutil.copyfileobj(rp, wp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prefix", type=str, dest="prefix", default="bag-of-words")
    ap.add_argument("-v", "--verbose", type=int, dest="verbose", default=0)

    main(**vars(ap.parse_args()))
