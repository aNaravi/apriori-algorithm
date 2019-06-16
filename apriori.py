import os
import argparse
import json
import datetime as dt
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import lil_array

from utilities import jsonEncoder, Timer

CHUNK = 10000000

def generate_itemset_candidates(frequent_itemsets):
    for i, itemset in enumerate(frequent_itemsets):
        itemset_pruned = itemset[:-1]
        for next_itemset in frequent_itemsets[i + 1:]:
            next_itemset_pruned, last_item = next_itemset[:-1], next_itemset[-1]
            if itemset_pruned == next_itemset_pruned:
                yield [*itemset, last_item]
            else:
                break


def generate_rule_candidates(itemset, H):
    if not H:
        itemset = set(itemset)
        for consequent, in combinations(itemset, 1):
            yield [consequent], [item for item in itemset if item != consequent]
        return

    consequents = [c for (c, _) in H]

    for i, (consequent, antecedent) in enumerate(H):
        consequent_pruned = consequent[:-1]
        for next_consequent in consequents[i + 1:]:
            next_consequent_pruned, last_item = next_consequent[:-1], next_consequent[-1]
            if consequent_pruned == next_consequent_pruned:
                candidate = [*consequent, last_item]
                if all(set(c).issubset(itemset) for c in combinations(candidate, len(candidate) - 1)):
                    yield (candidate, list(item for item in antecedent if item != last_item))
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


def association_rules(frequent_itemsets: 'list[list[int]]', K, minconf, ntransactions, MX, vocab):
    association_rules: list[str] = []

    for itemset in frequent_itemsets:
        H_m = list()
        itemset_support: float = (MX[:, itemset].sum(axis=1) >= len(itemset)).sum() / ntransactions

        m = 0
        while ((m := m + 1) < K) and (len(H := H_m) > 0 or m < 2):
            H_m = list()
            for (consequent, antecedent) in generate_rule_candidates(itemset, H):
                antecedent_support = (MX[:, antecedent].sum(axis=1) >= len(antecedent)).sum() / ntransactions
                if (confidence := itemset_support / antecedent_support) >= minconf:
                    association_rules.append(f"C:{confidence:.2f} S:{itemset_support:.2f}  {vocab[antecedent]} -> {vocab[consequent]}")
                    H_m.append((consequent, antecedent))

    return association_rules


def main(K: int, minsup: float, minconf: float, collection: str, prefix: str):
    with Timer("Adjacency Matrix") as t1:
        with open(f"{prefix}/docword.{collection}.txt") as file:
            D, W = [int(next(file).split('\\')[0]) for _ in range(2)]

        vocab = pd.read_csv(f"{prefix}/vocab.{collection}.txt", sep='\t', names=["words"]).squeeze("columns")

        docword_len = 0
        MX = lil_array((D, W), dtype=np.int8)
        with pd.read_csv(f"{prefix}/docword.{collection}.txt", skiprows=3, sep=' ', names=['docID', 'wordID'], usecols=[0, 1], chunksize=CHUNK) as reader:
            for docword in reader:
                MX[docword['docID'].values - 1, docword['wordID'].values - 1] = 1
                docword_len += len(docword)
            del docword
        MX = MX.tocsc()

    print(f"{t1} \t MX: {(MX.data.nbytes + MX.indices.nbytes + MX.indptr.nbytes) * 8 / (1000 ** 2):.2f}MB  docword: {docword_len}")

    with Timer("Frequent Itemsets") as t2:
        F_K = frequent_itemsets(K, minsup, D, MX)

    print(f"{t2} \t {len(F_K)}")

    with Timer("Association Rules") as t3:
        H_K = association_rules(F_K, K, minconf, D, MX, vocab.values)

    print(f"{t3} \t {len(H_K)}")

    with Timer("Result") as t4:
        result = {
            "frequent_itemsets": vocab.values[F_K],
            "association_rules": H_K
        }

        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/apriori_{dt.datetime.now():%y-%m-%d_%H-%M-%S}_{collection}_K{K}_S{minsup}_C{minconf}.json", 'w') as fp:
            json.dump(result, fp, indent=4, cls=jsonEncoder)

    print(f"{t4}")


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

    def minconf_type(var):
        try:
            var = float(var)
            if not 0 < var < 1:
                raise argparse.ArgumentTypeError("minconf must be in (0, 1)")
            else:
                return var
        except TypeError:
            raise argparse.ArgumentTypeError("minconf must be in (0, 1)")


    ap = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    ap.add_argument("K", type=int)
    ap.add_argument("minsup", type=minsup_type)
    ap.add_argument("minconf", type=minconf_type)
    ap.add_argument("collection", type=str, choices=["enron", "kos", "nips", "nytimes", "pubmed"])
    ap.add_argument("-p", "--prefix", type=str, dest="prefix", default="bag-of-words", help="dataset location (default: bag-of-words)")

    main(**vars(ap.parse_args()))
