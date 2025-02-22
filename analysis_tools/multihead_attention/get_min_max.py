

import numpy as np

def get_min_max_QKV_matrix(QKV_list, epochs_to_analyze, N_attention_layers):
    # Initialization to extreme values
    min_val = 1e9
    max_val = -1e9

    for epoch in range(len(epochs_to_analyze)):
        for atten_layer in range(N_attention_layers):
            list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['query'])
            min_val = min(min_val, np.min(list_vals))
            max_val = max(max_val, np.max(list_vals))

            list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['key'])
            min_val = min(min_val, np.min(list_vals))
            max_val = max(max_val, np.max(list_vals))

            list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['value'])
            min_val = min(min_val, np.min(list_vals))
            max_val = max(max_val, np.max(list_vals))

    return min_val, max_val

def get_min_max_QKV_head(QKV_list, epochs_to_analyze, N_attention_layers, entropy = True):
    # Initialization to extreme values
    min_val = 1e9
    max_val = -1e9

    for epoch in range(len(epochs_to_analyze)):
        for atten_layer in range(N_attention_layers):
            sel_str = 'query' if entropy else 'qk_mi'
            list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
            for list_vals in list_of_lists:
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

            sel_str = 'key' if entropy else 'qv_mi'
            list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
            for list_vals in list_of_lists:
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

            sel_str = 'value' if entropy else 'kv_mi'
            list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
            for list_vals in list_of_lists:
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

    return min_val, max_val
