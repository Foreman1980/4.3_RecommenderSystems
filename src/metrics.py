import pandas as pd
import numpy as np


def precision_at_k(recommended_list, bought_list, k=5):
    return np.isin(bought_list, recommended_list[:k]).sum() / len(recommended_list[:k])
    #   сделать в домашней работе
    # if k <= len(recommended_list):
    #     return np.isin(bought_list, recommended_list[:k]).sum() / k
    # else:
    #     raise ValueError(f'k = {k}, which exceeds the length of the recommendation list = {len(recommended_list)}')


def recall_at_k(recommended_list, bought_list, k=5):
    return np.isin(bought_list, recommended_list[:k]).sum() / len(bought_list)


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0

    amount_relevant = len(relevant_indexes)

    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant


def map_k(recommended_list, bought_list, k=5):
    n = len(recommended_list)
    return sum([ap_k(recommended_list[i], bought_list[i], k) for i in range(n)]) / n


from math import log


def ndcg_at_k(recommended_list, bought_list, k=5):
    def dcg_k(flags, k):
        def solve(i):
            if i <= 2:
                return i
            else:
                return log(i)

        return sum([flags[i] / solve(i + 1) for i in range(k)]) / k

    if len(recommended_list) == 0:
        return 0.
    elif len(recommended_list) < k:
        k = len(recommended_list)
    flags = np.isin(recommended_list[:k], bought_list)
    return dcg_k(flags, k) / dcg_k([1] * k, k)


def reciprocal_rank(recommended_list, bought_list, k=5):
    #   сделать в домашней работе
    relevant_indexes = np.nonzero(np.isin(recommended_list[:k], bought_list))[0]
    if len(relevant_indexes) > 0:
        return 1 / (relevant_indexes[0] + 1)
    else:
        return 0


def mrr_at_k(recommended_lists, bought_lists, k=5):
    n = len(recommended_lists)
    return sum([reciprocal_rank(recommended_lists[i], bought_lists[i], k) for i in range(n)]) / n
