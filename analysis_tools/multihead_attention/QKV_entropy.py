
import sys
sys.path.append('../utils')

import numpy as np
from mutual_info_estimator import MutualInfoEstimator
from tqdm import tqdm


def compute_entropy(query, key, value, N_dimensions) -> dict:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    # These will contain entropy for the query, key and value arrays
    # for each attention layer
    Q_entropy_list = list()
    K_entropy_list = list()
    V_entropy_list = list()

    for n in tqdm(range(N_dimensions)):
        Y = query[:, n]
        MI_estimator.set_inputs(Y, Y)
        # H = MI_estimator.kraskov_entropy()
        prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        H = mi_dict["H_X"]
        Q_entropy_list.append(H)

        Y = key[:, n]
        MI_estimator.set_inputs(Y, Y)
        # H = MI_estimator.kraskov_entropy()
        prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        H = mi_dict["H_X"]
        K_entropy_list.append(H)

        Y = value[:, n]
        MI_estimator.set_inputs(Y, Y)
        # H = MI_estimator.kraskov_entropy()
        prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=max(100, len(Y)))
        H = mi_dict["H_X"]
        V_entropy_list.append(H)

    entropy_dict = dict(Q_entropy_list=Q_entropy_list, K_entropy_list=K_entropy_list, V_entropy_list=V_entropy_list)

    return entropy_dict

def compute_QKV_matrix_entropy(QKV_dict : dict) -> dict:
    # This will contain the entropy values for the query, key and value arrays
    QKV_entropy_dict = dict()

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

        entropy_dict = compute_entropy(query, key, value, N_dimensions)

        Q_entropy_list = entropy_dict["Q_entropy_list"]
        K_entropy_list = entropy_dict["K_entropy_list"]
        V_entropy_list = entropy_dict["V_entropy_list"]

        QKV_entropy_dict[f'attention_{i}'] = {"query": Q_entropy_list, "key": K_entropy_list, "value": V_entropy_list}

    return QKV_entropy_dict
