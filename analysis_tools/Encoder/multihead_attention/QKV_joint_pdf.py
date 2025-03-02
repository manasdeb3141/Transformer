
import sys
sys.path.append('../utils')

import numpy as np
from mutual_info_estimator import MutualInfoEstimator
from tqdm import tqdm
import time

def compute_x_entropy(x, query, key, value, N_dimensions) -> dict:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    # These will contain entropy for the input over dimensions
    # for each attention layer
    x_entropy_list = list()
    q_entropy_list = list()
    k_entropy_list = list()
    v_entropy_list = list()
    xq_entropy_list = list()
    xk_entropy_list = list()
    xv_entropy_list = list()

    for n in tqdm(range(N_dimensions)):
        # Query
        X = x[:, n]
        Y = query[:, n]
        MI_estimator.set_inputs(X, Y)
        _, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
        x_entropy_list.append(mi_dict["H_X"])
        q_entropy_list.append(mi_dict["H_Y"])
        xq_entropy_list.append(mi_dict["H_XY"])

        # Key
        X = x[:, n]
        Y = key[:, n]
        MI_estimator.set_inputs(X, Y)
        _, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
        k_entropy_list.append(mi_dict["H_Y"])
        xk_entropy_list.append(mi_dict["H_XY"])

        # Value
        X = x[:, n]
        Y = value[:, n]
        MI_estimator.set_inputs(X, Y)
        _, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
        v_entropy_list.append(mi_dict["H_Y"])
        xv_entropy_list.append(mi_dict["H_XY"])

    entropy_dict = dict(x_entropy_list=x_entropy_list, 
                        q_entropy_list=q_entropy_list, 
                        k_entropy_list=k_entropy_list, 
                        v_entropy_list=v_entropy_list,
                        xq_entropy_list=xq_entropy_list, 
                        xk_entropy_list=xk_entropy_list, 
                        xv_entropy_list=xv_entropy_list)

    return entropy_dict

def compute_joint_pdf(x, query, key, value) -> dict:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    x_in = x.reshape(-1, 1)
    q_prime = query.reshape(-1, 1)
    k_prime = key.reshape(-1, 1)
    v_prime = value.reshape(-1, 1)

    # Compute joint PDF and MI between x and q_prime
    print("Computing the joint PDF and MI between x and q_prime ...")
    start_time = time.time()
    MI_estimator.set_inputs(x_in, q_prime)
    q_prob_dict, q_mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
    print(f"Time taken: {time.time() - start_time}")

    # Compute joint PDF and MI between x and k_prime
    print("\nComputing the joint PDF and MI between x and k_prime ...")
    start_time = time.time()
    MI_estimator.set_inputs(x_in, k_prime)
    k_prob_dict, k_mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
    print(f"Time taken: {time.time() - start_time}")

    # Compute joint PDF and MI between x and v_prime
    print("\nComputing the joint PDF and MI between x and v_prime ...")
    start_time = time.time()
    MI_estimator.set_inputs(x_in, v_prime)
    v_prob_dict, v_mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
    print(f"Time taken: {time.time() - start_time}")

    return dict(Wq_joint_pdf=q_prob_dict, Wk_joint_pdf=k_prob_dict, Wv_joint_pdf=v_prob_dict), \
            dict(x_q_mi=q_mi_dict, x_k_mi=k_mi_dict, x_v_mi=v_mi_dict)

def compute_QKV_joint_pdf(QKV_stacked_dict : dict, attention_layer : int, N_tokens : int) -> dict:
    # This will contain the joint pdf for the query, key and value arrays
    QKV_joint_pdf_dict = dict()

    # Get the dimensions of the model from the number
    # of columns of the input array
    x = QKV_stacked_dict[f"attention_{attention_layer}"]["x"]
    N_dimensions = x.shape[1]

    print(f"Computing the joint PDF for attention layer {attention_layer} ...")
    query = QKV_stacked_dict[f'attention_{attention_layer}']["query"]
    key = QKV_stacked_dict[f'attention_{attention_layer}']["key"]
    value = QKV_stacked_dict[f'attention_{attention_layer}']["value"]
    joint_pdf_dict, mi_dict = compute_joint_pdf(x, query, key, value)

    Wq_joint_pdf = joint_pdf_dict["Wq_joint_pdf"]
    Wk_joint_pdf = joint_pdf_dict["Wk_joint_pdf"]
    Wv_joint_pdf = joint_pdf_dict["Wv_joint_pdf"]

    x_q_mi = mi_dict["x_q_mi"]
    x_k_mi = mi_dict["x_k_mi"]
    x_v_mi = mi_dict["x_v_mi"] 

    compute_entropy = True
    x_token_entropy_list = list()
    q_token_entropy_list = list()
    k_token_entropy_list = list()
    v_token_entropy_list = list()
    xq_token_entropy_list = list()
    xk_token_entropy_list = list()
    xv_token_entropy_list = list()
    if compute_entropy:
        for _ in range(N_tokens):
            x_entropy_dict = compute_x_entropy(x, query, key, value, N_dimensions)
            x_entropy_list = x_entropy_dict["x_entropy_list"]
            q_entropy_list = x_entropy_dict["q_entropy_list"]
            k_entropy_list = x_entropy_dict["k_entropy_list"]
            v_entropy_list = x_entropy_dict["v_entropy_list"]
            xq_entropy_list = x_entropy_dict["xq_entropy_list"]
            xk_entropy_list = x_entropy_dict["xk_entropy_list"]
            xv_entropy_list = x_entropy_dict["xv_entropy_list"]

            x_token_entropy_list.append(x_entropy_list)
            q_token_entropy_list.append(q_entropy_list)
            k_token_entropy_list.append(k_entropy_list)
            v_token_entropy_list.append(v_entropy_list)
            xq_token_entropy_list.append(xq_entropy_list)
            xk_token_entropy_list.append(xk_entropy_list)
            xv_token_entropy_list.append(xv_entropy_list)
    else:
        x_token_entropy_list = q_token_entropy_list = k_token_entropy_list = v_token_entropy_list = xq_token_entropy_list = xk_token_entropy_list = xv_token_entropy_list = list()

    QKV_joint_pdf_dict[f'attention_{attention_layer}'] = \
        {"Wq_joint_pdf": Wq_joint_pdf, "Wk_joint_pdf": Wk_joint_pdf, "Wv_joint_pdf": Wv_joint_pdf, 
         "x_q_mi": x_q_mi, "x_k_mi": x_k_mi, "x_v_mi": x_v_mi,
         "x_entropy_list": x_token_entropy_list, "q_entropy_list": q_token_entropy_list, "k_entropy_list": k_token_entropy_list,
         "v_entropy_list": v_token_entropy_list, "xq_entropy_list": xq_token_entropy_list, "xk_entropy_list": xk_token_entropy_list,
         "xv_entropy_list": xv_token_entropy_list}
                                       
    return QKV_joint_pdf_dict