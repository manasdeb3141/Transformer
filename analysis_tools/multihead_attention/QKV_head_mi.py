    

import sys
sys.path.append('../utils')

import numpy as np
from mutual_info_estimator import MutualInfoEstimator

def QKV_head_mi(query_head, key_head, value_head, seq_len) -> dict:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    # Contains the mutual information of the pairwise query, key and value arrays
    # for each token in the sequence
    query_key_MI_list = list()
    query_value_MI_list = list()
    key_value_MI_list = list()

    for i in range(seq_len):
        X = query_head[i]
        Y = key_head[i]
        MI_estimator.set_inputs(X, Y)
        # MI = MI_estimator.kraskov_MI()["MI"]
        MI = MI_estimator.MINE_MI()
        query_key_MI_list.append(MI)

        X = query_head[i]
        Y = value_head[i]
        MI_estimator.set_inputs(X, Y)
        # MI = MI_estimator.kraskov_MI()["MI"]
        MI = MI_estimator.MINE_MI()
        query_value_MI_list.append(MI)

        X = key_head[i]
        Y = value_head[i]
        MI_estimator.set_inputs(X, Y)
        # MI = MI_estimator.kraskov_MI()["MI"]
        MI = MI_estimator.MINE_MI()
        key_value_MI_list.append(MI)

    mi_dict = dict(QK_mi_list=query_key_MI_list, QV_mi_list=query_value_MI_list, KV_mi_list=key_value_MI_list)
    return mi_dict  


def compute_QKV_head_mi(QKV_dict : dict, N_attention_layers : int):
    # This will contain the entropy values for the query, key and value arrays
    QKV_entropy_dict = dict()

    # Get the dimensions of the model from the number
    # of columns of the query array
    query = QKV_dict['attention_0']["query"]
    N_heads = query.shape[0]
    seq_len = query.shape[1]

    for i in range(N_attention_layers):
        query = QKV_dict[f'attention_{i}']["query"]
        key = QKV_dict[f'attention_{i}']["key"]
        value = QKV_dict[f'attention_{i}']["value"]

        # Contains a list of lists. The number of elements 
        # in each of the following lists is equal to the 
        # number of attention heads. Each element in the list
        # is a list containing the mutual information across the
        # seq_len dimension of the corresponding query and key heads
        QK_head_mi_list = list()
        QV_head_mi_list = list()
        KV_head_mi_list = list()

        for j in range(N_heads):
            query_head = query[j]
            key_head = key[j]
            value_head = value[j]

            mi_dict = QKV_head_mi(query_head, key_head, value_head, seq_len)

            # The dictionary contains the list of mutual information values between the query and key 
            # attention head arrays across the seq_len dimension
            QK_head_mi_list.append(mi_dict["QK_mi_list"])
            QV_head_mi_list.append(mi_dict["QV_mi_list"])
            KV_head_mi_list.append(mi_dict["KV_mi_list"])

        QKV_entropy_dict[f'attention_{i}'] = {"qk_mi": QK_head_mi_list, "qv_mi": QV_head_mi_list, "kv_mi": KV_head_mi_list}

    return QKV_entropy_dict
