import argparse
import numpy as np
import pandas as pd
from scipy.sparse import lil_array

CHUNK = 100000000

def generate_frequent_itemsets(K: int, minsup: float, collection: str, prefix: str):

    def candidates_generator(frequent_itemsets):
        for i, itemset in enumerate(frequent_itemsets):
            itemset_pruned = itemset[:-1]
            for next_itemset in frequent_itemsets[i + 1:]:
                next_itemset_pruned, last_item = next_itemset[:-1], next_itemset[-1]
                if np.array_equal(itemset_pruned, next_itemset_pruned):
                    yield np.append(itemset, last_item)
                else:
                    break


    with open(f"{prefix}/docword.{collection}.txt") as file:
        D, W = [int(next(file).split('\\')[0]) for _ in range(2)]

    MX = lil_array((D, W), dtype=np.int8)
    with pd.read_csv(f"{prefix}/docword.{collection}.txt", skiprows=3, sep=' ', names=['docID', 'wordID'], usecols=[0, 1], iterator=True) as reader:
        docword = reader.get_chunk(CHUNK)
        MX[docword['docID'].values - 1, docword['wordID'].values - 1] = 1
    del docword

    MX = MX.tocsc()
    k = 1
    F_k = np.flatnonzero(MX.sum(axis=0) >= D * minsup).reshape(-1,1)

    while (frequent_itemsets := F_k).size > 0 and (k := k + 1) <= K:
        F_k = np.empty((0, k), dtype=np.int32)
        for candidate in candidates_generator(frequent_itemsets):
            if (MX[:, candidate].sum(axis=1) >= len(candidate)).sum() >= D * minsup:
                F_k = np.append(F_k, candidate.reshape(1,-1), axis=0)
    del MX

    vocab = pd.read_csv(f"{prefix}/vocab.{collection}.txt", names=["words"]).squeeze("columns")

    return vocab.values[frequent_itemsets]
