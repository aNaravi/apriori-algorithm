import os
from tempfile import mkdtemp
import numpy as np
import pandas as pd


def generate_frequent_itemsets(collection, K, F, **kwargs):

    def candidates_generator(frequent_itemsets):
        for i, itemset in enumerate(frequent_itemsets):
            itemset_pruned = itemset[:-1]
            for next_itemset in frequent_itemsets[i + 1:]:
                next_itemset_pruned, last_item = next_itemset[:-1], next_itemset[-1]
                if np.array_equal(itemset_pruned, next_itemset_pruned):
                    yield np.append(itemset, last_item)
                else:
                    break

    max_chunk_size = kwargs.get("max_chunk_size", 50000000)

    with open('docword.' + collection + '.txt') as file:
        D, W = [int(next(file).split('\\')[0]) for n in range(2)]

    try:
        MX = np.zeros((D,W), dtype=np.int8)
    except MemoryError:
        filename = os.path.join(mkdtemp(), 'MX.dat')
        MX = np.memmap(filename, dtype=np.int8, mode='w+', shape=(D,W))
        for docword in pd.read_csv('docword.' + collection + '.txt', skiprows=3, sep=' ',
                                   names=['docID', 'wordID'], usecols=[0, 1],
                                   chunksize=max_chunk_size):
            MX[docword.docID.values - 1, docword.wordID.values - 1] = 1
        del docword
    else:
        docword = pd.read_csv('docword.' + collection + '.txt', skiprows=3, sep=' ', names=['docID', 'wordID'], usecols=[0, 1])
        MX[docword.docID.values - 1,docword.wordID.values - 1] = 1

    frequent_itemsets = np.where(MX.sum(axis=0) >= D * F)[0].reshape(-1,1)
    i = 2
    while i <= K and frequent_itemsets.size > 0:
        freq = np.empty((0,i), dtype=np.int)
        for c in candidates_generator(frequent_itemsets):
            if np.all(MX[:, c], axis=1).sum() >= D * F:
                freq = np.append(freq, c.reshape(1,-1), axis=0)
        frequent_itemsets = freq
        i += 1

    if type(MX) == np.core.memmap:
        del MX
        os.remove(filename)
        os.removedirs(os.path.dirname(filename))
        print("Deleted ", filename)

    vocab = pd.read_csv('vocab.' + collection + '.txt', names=['words'])

    return vocab.words.values[frequent_itemsets]
