
import sys
sys.path.append('../utils')

import numpy as np
from mutual_info_estimator import MutualInfoEstimator
from tqdm import tqdm


def compute_entropy_mi(x, query, key, value, N_dimensions) -> dict:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    kernel_MI = True

    # These will contain entropy for the query, key and value arrays
    # for each attention layer
    Q_entropy_list = list()
    K_entropy_list = list()
    V_entropy_list = list()

    Q_mi_list = list()
    K_mi_list = list()
    V_mi_list = list()

    for n in tqdm(range(N_dimensions)):
        # Query
        X = x[:, n]
        Y = query[:, n]
        MI_estimator.set_inputs(X, Y)

        if kernel_MI:
            prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        else:
            mi_dict = MI_estimator.kraskov_MI()

        H = mi_dict["H_Y"]
        Q_entropy_list.append(H)
        MI = mi_dict["MI"]
        Q_mi_list.append(MI)

        # Key
        X = x[:, n]
        Y = key[:, n]
        MI_estimator.set_inputs(X, Y)

        if kernel_MI:
            prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        else:
            mi_dict = MI_estimator.kraskov_MI()

        H = mi_dict["H_Y"]
        K_entropy_list.append(H)
        MI = mi_dict["MI"]
        K_mi_list.append(MI)

        # Value
        X = x[:, n]
        Y = value[:, n]
        MI_estimator.set_inputs(X, Y)

        if kernel_MI:
            prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        else:
            mi_dict = MI_estimator.kraskov_MI()

        H = mi_dict["H_Y"]
        V_entropy_list.append(H)
        MI = mi_dict["MI"]
        V_mi_list.append(MI)

    entropy_mi_dict = dict(Q_entropy_list=Q_entropy_list, K_entropy_list=K_entropy_list, V_entropy_list=V_entropy_list,
                        Q_mi_list=Q_mi_list, K_mi_list=K_mi_list, V_mi_list=V_mi_list)

    return entropy_mi_dict

def compute_QKV_matrix_entropy_mi(QKV_dict : dict) -> dict:
    # This will contain the entropy values for the query, key and value arrays
    QKV_entropy_mi_dict = dict()

    # Get the dimensions of the model from the number
    # of columns of the query array
    x = QKV_dict['attention_0']["x"]
    N_dimensions = x.shape[1]

    N_attention_layers = len(QKV_dict)    

    for i in range(N_attention_layers):
        print(f"Computing the entropy for attention layer {i} ...")
        x = QKV_dict[f'attention_{i}']["x"]
        query = QKV_dict[f'attention_{i}']["query"]
        key = QKV_dict[f'attention_{i}']["key"]
        value = QKV_dict[f'attention_{i}']["value"]

        entropy_dict = compute_entropy_mi(x, query, key, value, N_dimensions)

        Q_entropy_list = entropy_dict["Q_entropy_list"]
        K_entropy_list = entropy_dict["K_entropy_list"]
        V_entropy_list = entropy_dict["V_entropy_list"]
        Q_mi_list = entropy_dict["Q_mi_list"]
        K_mi_list = entropy_dict["K_mi_list"]
        V_mi_list = entropy_dict["V_mi_list"]

        QKV_entropy_mi_dict[f'attention_{i}'] = {"query_entropy": Q_entropy_list, "key_entropy": K_entropy_list, "value_entropy": V_entropy_list,
                                              "query_mi": Q_mi_list, "key_mi": K_mi_list, "value_mi": V_mi_list}

    return QKV_entropy_mi_dict
