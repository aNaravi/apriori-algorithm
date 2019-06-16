import argparse
import numpy as np
import pandas as pd
from scipy.sparse import lil_array

CHUNK = 100000000

def generate_itemset_candidates(frequent_itemsets):
    for i, itemset in enumerate(frequent_itemsets):
        itemset_pruned = itemset[:-1]
        for next_itemset in frequent_itemsets[i + 1:]:
            next_itemset_pruned, last_item = next_itemset[:-1], next_itemset[-1]
            if itemset_pruned == next_itemset_pruned:
                yield [*itemset, last_item]
            else:
                break



def frequent_itemsets(K: int, minsup: float, ntransactions: int, MX):
    k = 1
    F_k = np.flatnonzero(MX.sum(axis=0) >= ntransactions * minsup).reshape(-1,1).tolist()

    while (len(frequent_itemsets := F_k) > 0) and ((k := k + 1) <= K):
        F_k = list()
        for candidate in generate_itemset_candidates(frequent_itemsets):
            if (MX[:, candidate].sum(axis=1) >= len(candidate)).sum() >= ntransactions * minsup:
                F_k.append(candidate)

    return frequent_itemsets


def main(K: int, minsup: float, minconf: float, collection: str, prefix: str):
    with open(f"{prefix}/docword.{collection}.txt") as file:
        D, W = [int(next(file).split('\\')[0]) for _ in range(2)]

    vocab = pd.read_csv(f"{prefix}/vocab.{collection}.txt", names=["words"]).squeeze("columns")

    MX = lil_array((D, W), dtype=np.int8)
    with pd.read_csv(f"{prefix}/docword.{collection}.txt", skiprows=3, sep=' ', names=['docID', 'wordID'], usecols=[0, 1], iterator=True) as reader:
        docword = reader.get_chunk(CHUNK)
        MX[docword['docID'].values - 1, docword['wordID'].values - 1] = 1
    del docword
    MX = MX.tocsc()

    F_K = frequent_itemsets(K, minsup, D, MX)
    print(f"{vocab.values[F_K]}")



if __name__ == "__main__":

    def minsup_type(var):
        try:
            var = float(var)
            if not 0 < var < 1:
                raise argparse.ArgumentTypeError("minsup must be in (0, 1)")
            else:
                return var
        except TypeError:
            raise argparse.ArgumentTypeError("minsup must be in (0, 1)")


    ap = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    ap.add_argument("K", type=int)
    ap.add_argument("minsup", type=minsup_type)
    ap.add_argument("collection", type=str, choices=["enron", "kos", "nips", "nytimes", "pubmed"])
    ap.add_argument("-p", "--prefix", type=str, dest="prefix", default="bag-of-words", help="dataset location (default: bag-of-words)")

    main(**vars(ap.parse_args()))
