

from QKV_entropy import compute_entropy

def compute_QKV_head_entropy(QKV_dict : dict, N_attention_layers : int) -> dict:
    # This will contain the entropy values for the query, key and value arrays
    QKV_entropy_dict = dict()

    # Get the dimensions of the model from the number
    # of columns of the query array
    query = QKV_dict['attention_0']["query"]
    N_heads = query.shape[0]
    N_dimensions = query.shape[2]

    for i in range(N_attention_layers):
        query = QKV_dict[f'attention_{i}']["query"]
        key = QKV_dict[f'attention_{i}']["key"]
        value = QKV_dict[f'attention_{i}']["value"]

        # Contains a list of lists. The number of elements 
        # in each of the following lists is equal to the 
        # number of attention heads. Each element in the list
        # is a list containing the entropy across dimensions 
        # of the corresponding head of the query, key or value array
        Q_head_entropy_list = list()
        K_head_entropy_list = list()
        V_head_entropy_list = list()

        for j in range(N_heads):
            query_head = query[j]
            key_head = key[j]
            value_head = value[j]

            entropy_dict = compute_entropy(query_head, key_head, value_head, N_dimensions)

            # The dictionary contains the list of entropy values for the query, key and value 
            # attention head arrays across all dimensions
            Q_head_entropy_list.append(entropy_dict["Q_entropy_list"])
            K_head_entropy_list.append(entropy_dict["K_entropy_list"])
            V_head_entropy_list.append(entropy_dict["V_entropy_list"])

        QKV_entropy_dict[f'attention_{i}'] = {"query": Q_head_entropy_list, "key": K_head_entropy_list, "value": V_head_entropy_list}

    return QKV_entropy_dict

