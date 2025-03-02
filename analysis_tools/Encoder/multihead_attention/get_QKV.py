
import sys
sys.path.append('../..')
sys.path.append('../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_query_key_value_matrix(analyzer : TransformerAnalyzer, sentence_id : int, N_src_tokens : int):
    # Attention layer 0
    enc_attn_input = analyzer.enc_0_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_0 = x[:N_src_tokens]
    query_0 = query[:N_src_tokens]
    key_0 = key[:N_src_tokens]
    value_0 = value[:N_src_tokens]

    # Attention layer 1
    enc_attn_input = analyzer.enc_1_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_1 = x[:N_src_tokens]
    query_1 = query[:N_src_tokens]
    key_1 = key[:N_src_tokens]
    value_1 = value[:N_src_tokens]
    
    # Attention layer 2
    enc_attn_input = analyzer.enc_2_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_2 = x[:N_src_tokens]
    query_2 = query[:N_src_tokens]
    key_2 = key[:N_src_tokens]
    value_2 = value[:N_src_tokens]

    # Attention layer 3
    enc_attn_input = analyzer.enc_3_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_3 = x[:N_src_tokens]
    query_3 = query[:N_src_tokens]
    key_3 = key[:N_src_tokens]
    value_3 = value[:N_src_tokens]

    # Attention layer 4
    enc_attn_input = analyzer.enc_4_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_4 = x[:N_src_tokens]
    query_4 = query[:N_src_tokens]
    key_4 = key[:N_src_tokens]
    value_4 = value[:N_src_tokens]

    # Attention layer 5
    enc_attn_input = analyzer.enc_5_attn_probe._probe_in[sentence_id]
    x = enc_attn_input["q"].squeeze()
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    x_5 = x[:N_src_tokens]
    query_5 = query[:N_src_tokens]
    key_5 = key[:N_src_tokens]
    value_5 = value[:N_src_tokens]

    QKV_dict = { 'attention_0': {"x": x_0, "query": query_0, "key": key_0, "value": value_0},
                 'attention_1': {"x": x_1, "query": query_1, "key": key_1, "value": value_1},
                 'attention_2': {"x": x_2, "query": query_2, "key": key_2, "value": value_2},
                 'attention_3': {"x": x_3, "query": query_3, "key": key_3, "value": value_3},
                 'attention_4': {"x": x_4, "query": query_4, "key": key_4, "value": value_4},
                 'attention_5': {"x": x_5, "query": query_5, "key": key_5, "value": value_5} }

    return QKV_dict 


def get_query_key_value_head(analyzer : TransformerAnalyzer, sentence_id : int, N_src_tokens : int = None):
    # Attention layer 0
    enc_attn_input = analyzer.enc_0_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()

    if N_src_tokens is None:
        N_src_tokens = query.shape[0]

    query_0 = query[:,:N_src_tokens]
    key_0 = key[:,:N_src_tokens]
    value_0 = value[:,:N_src_tokens]

    # Attention layer 1
    enc_attn_input = analyzer.enc_1_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()
    query_1 = query[:,:N_src_tokens]
    key_1 = key[:,:N_src_tokens]
    value_1 = value[:,:N_src_tokens]
    
    # Attention layer 2
    enc_attn_input = analyzer.enc_2_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()
    query_2 = query[:,:N_src_tokens]
    key_2 = key[:,:N_src_tokens]
    value_2 = value[:,:N_src_tokens]

    # Attention layer 3
    enc_attn_input = analyzer.enc_3_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()
    query_3 = query[:,:N_src_tokens]
    key_3 = key[:,:N_src_tokens]
    value_3 = value[:,:N_src_tokens]

    # Attention layer 4
    enc_attn_input = analyzer.enc_4_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()
    query_4 = query[:,:N_src_tokens]
    key_4 = key[:,:N_src_tokens]
    value_4 = value[:,:N_src_tokens]

    # Attention layer 5
    enc_attn_input = analyzer.enc_5_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query_head"].squeeze()
    key = enc_attn_input["key_head"].squeeze()
    value = enc_attn_input["value_head"].squeeze()
    query_5 = query[:,:N_src_tokens]
    key_5 = key[:,:N_src_tokens]
    value_5 = value[:,:N_src_tokens]

    QKV_dict = { 'attention_0': {"query": query_0, "key": key_0, "value": value_0},
                 'attention_1': {"query": query_1, "key": key_1, "value": value_1},
                 'attention_2': {"query": query_2, "key": key_2, "value": value_2},
                 'attention_3': {"query": query_3, "key": key_3, "value": value_3},
                 'attention_4': {"query": query_4, "key": key_4, "value": value_4},
                 'attention_5': {"query": query_5, "key": key_5, "value": value_5} }

    return QKV_dict 