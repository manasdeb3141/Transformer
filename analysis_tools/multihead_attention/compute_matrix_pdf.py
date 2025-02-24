   

import sys
sys.path.append('../utils')

import numpy as np
from mutual_info_estimator import MutualInfoEstimator
from reduce_dimensions import reduce_dimensions
from tqdm import tqdm


def compute_ff_pdf(ff_stacked_dict : dict, attention_layer : int):
    match attention_layer:
        case 0:
            ff_in = ff_stacked_dict["ff_0"]["ff_in"]
            ff_out = ff_stacked_dict["ff_0"]["ff_out"]

        case 1:
            ff_in = ff_stacked_dict["ff_1"]["ff_in"]
            ff_out = ff_stacked_dict["ff_1"]["ff_out"]

        case 2:
            ff_in = ff_stacked_dict["ff_2"]["ff_in"]
            ff_out = ff_stacked_dict["ff_2"]["ff_out"]

        case 3:
            ff_in = ff_stacked_dict["ff_3"]["ff_in"]
            ff_out = ff_stacked_dict["ff_3"]["ff_out"]

        case 4:
            ff_in = ff_stacked_dict["ff_4"]["ff_in"]
            ff_out = ff_stacked_dict["ff_4"]["ff_out"]

        case 5:
            ff_in = ff_stacked_dict["ff_5"]["ff_in"]
            ff_out = ff_stacked_dict["ff_5"]["ff_out"]

        case 6:
            ff_in = ff_stacked_dict["ff_6"]["ff_in"]
            ff_out = ff_stacked_dict["ff_6"]["ff_out"]

    # X_reduced, Y_reduced = reduce_dimensions(ff_in, ff_out)
    # Convert the arrays to column vectors
    # X = X_reduced.reshape(-1, 1)
    # Y = Y_reduced.reshape(-1, 1)

    column_MI = True

    if column_MI == True:
        N_dimensions = ff_in.shape[1]
        MI_data_list = list()
        for col in tqdm(range(N_dimensions)):
            X = ff_in[:, col]
            Y = ff_out[:, col]
        
            # Instantiate the Mutual Information Estimator object
            MI_estimator = MutualInfoEstimator(X, Y)
            MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
            MI_data_list.append(MI_data)
    else:
        seq_len = ff_in.shape[0]
        MI_data_list = list()
        for row in tqdm(range(seq_len)):
            X = ff_in[row, :]
            Y = ff_out[row, :]
        
            # Instantiate the Mutual Information Estimator object
            MI_estimator = MutualInfoEstimator(X, Y)
            MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
            MI_data_list.append(MI_data)


    return MI_data_list