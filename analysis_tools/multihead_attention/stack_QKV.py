
import numpy as np

def stack_QKV_matrix(QKV_list, N_inputs):
    # This will contain the query, key and value arrays
    # of all the input sentences of this epoch
    x_0_array = None; query_0_array = None;  key_0_array = None;  value_0_array = None
    x_1_array = None; query_1_array = None;  key_1_array = None;  value_1_array = None
    x_2_array = None; query_2_array = None;  key_2_array = None;  value_2_array = None
    x_3_array = None; query_3_array = None;  key_3_array = None;  value_3_array = None
    x_4_array = None; query_4_array = None;  key_4_array = None;  value_4_array = None
    x_5_array = None; query_5_array = None;  key_5_array = None;  value_5_array = None

    for i in range(N_inputs):
        x_0 = QKV_list[i]['attention_0']["x"]
        query_0 = QKV_list[i]['attention_0']["query"]
        key_0 = QKV_list[i]['attention_0']["key"]
        value_0 = QKV_list[i]['attention_0']["value"]
        x_1 = QKV_list[i]['attention_1']["x"]
        query_1 = QKV_list[i]['attention_1']["query"]
        key_1 = QKV_list[i]['attention_1']["key"]
        value_1 = QKV_list[i]['attention_1']["value"]
        x_2 = QKV_list[i]['attention_2']["x"]
        query_2 = QKV_list[i]['attention_2']["query"]
        key_2 = QKV_list[i]['attention_2']["key"]
        value_2 = QKV_list[i]['attention_2']["value"]
        x_3 = QKV_list[i]['attention_3']["x"]
        query_3 = QKV_list[i]['attention_3']["query"]
        key_3 = QKV_list[i]['attention_3']["key"]
        value_3 = QKV_list[i]['attention_3']["value"]
        x_4 = QKV_list[i]['attention_4']["x"]
        query_4 = QKV_list[i]['attention_4']["query"]
        key_4 = QKV_list[i]['attention_4']["key"]
        value_4 = QKV_list[i]['attention_4']["value"]
        x_5 = QKV_list[i]['attention_5']["x"]
        query_5 = QKV_list[i]['attention_5']["query"]
        key_5 = QKV_list[i]['attention_5']["key"]
        value_5 = QKV_list[i]['attention_5']["value"]

        if query_0_array is None:
            x_0_array = x_0
            query_0_array = query_0
            key_0_array = key_0
            value_0_array = value_0
        else:
            x_0_array = np.vstack((x_0_array, x_0))
            query_0_array = np.vstack((query_0_array, query_0))
            key_0_array = np.vstack((key_0_array, key_0))
            value_0_array = np.vstack((value_0_array, value_0))

        if query_1_array is None:
            x_1_array = x_1
            query_1_array = query_1
            key_1_array = key_1
            value_1_array = value_1 
        else:
            x_1_array = np.vstack((x_1_array, x_1))
            query_1_array = np.vstack((query_1_array, query_1))
            key_1_array = np.vstack((key_1_array, key_1))
            value_1_array = np.vstack((value_1_array, value_1))

        if query_2_array is None:
            x_2_array = x_2
            query_2_array = query_2
            key_2_array = key_2
            value_2_array = value_2
        else:
            x_2_array = np.vstack((x_2_array, x_2))
            query_2_array = np.vstack((query_2_array, query_2))
            key_2_array = np.vstack((key_2_array, key_2))
            value_2_array = np.vstack((value_2_array, value_2))

        if query_3_array is None:
            x_3_array = x_3
            query_3_array = query_3
            key_3_array = key_3
            value_3_array = value_3
        else:
            x_3_array = np.vstack((x_3_array, x_3))
            query_3_array = np.vstack((query_3_array, query_3))
            key_3_array = np.vstack((key_3_array, key_3))
            value_3_array = np.vstack((value_3_array, value_3))

        if query_4_array is None:
            x_4_array = x_4
            query_4_array = query_4
            key_4_array = key_4
            value_4_array = value_4
        else:
            x_4_array = np.vstack((x_4_array, x_4))
            query_4_array = np.vstack((query_4_array, query_4))
            key_4_array = np.vstack((key_4_array, key_4))
            value_4_array = np.vstack((value_4_array, value_4))
            
        if query_5_array is None:
            x_5_array = x_5
            query_5_array = query_5
            key_5_array = key_5
            value_5_array = value_5
        else:
            x_5_array = np.vstack((x_5_array, x_5))
            query_5_array = np.vstack((query_5_array, query_5))
            key_5_array = np.vstack((key_5_array, key_5))
            value_5_array = np.vstack((value_5_array, value_5))

    QKV_dict = { 'attention_0': {"x": x_0_array, "query": query_0_array, "key": key_0_array, "value": value_0_array},
                 'attention_1': {"x": x_1_array, "query": query_1_array, "key": key_1_array, "value": value_1_array},
                 'attention_2': {"x": x_2_array, "query": query_2_array, "key": key_2_array, "value": value_2_array},
                 'attention_3': {"x": x_3_array, "query": query_3_array, "key": key_3_array, "value": value_3_array},
                 'attention_4': {"x": x_4_array, "query": query_4_array, "key": key_4_array, "value": value_4_array},
                 'attention_5': {"x": x_5_array, "query": query_5_array, "key": key_5_array, "value": value_5_array} }

    return QKV_dict 
