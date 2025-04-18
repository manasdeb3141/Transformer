
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_self_atten_QKV(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id: int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_0 = Q[:decoder_token_id+1]
    K_0 = K[:decoder_token_id+1]
    V_0 = V[:decoder_token_id+1]
    Q_prime_0 = Q_prime[:decoder_token_id+1]
    K_prime_0 = K_prime[:decoder_token_id+1]
    V_prime_0 = V_prime[:decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_1 = Q[:decoder_token_id+1]
    K_1 = K[:decoder_token_id+1]
    V_1 = V[:decoder_token_id+1]
    Q_prime_1 = Q_prime[:decoder_token_id+1]
    K_prime_1 = K_prime[:decoder_token_id+1]
    V_prime_1 = V_prime[:decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_2 = Q[:decoder_token_id+1]
    K_2 = K[:decoder_token_id+1]
    V_2 = V[:decoder_token_id+1]
    Q_prime_2 = Q_prime[:decoder_token_id+1]
    K_prime_2 = K_prime[:decoder_token_id+1]
    V_prime_2 = V_prime[:decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_3 = Q[:decoder_token_id+1]
    K_3 = K[:decoder_token_id+1]
    V_3 = V[:decoder_token_id+1]
    Q_prime_3 = Q_prime[:decoder_token_id+1]
    K_prime_3 = K_prime[:decoder_token_id+1]
    V_prime_3 = V_prime[:decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_4 = Q[:decoder_token_id+1]
    K_4 = K[:decoder_token_id+1]
    V_4 = V[:decoder_token_id+1]
    Q_prime_4 = Q_prime[:decoder_token_id+1]
    K_prime_4 = K_prime[:decoder_token_id+1]
    V_prime_4 = V_prime[:decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_5 = Q[:decoder_token_id+1]
    K_5 = K[:decoder_token_id+1]
    V_5 = V[:decoder_token_id+1]
    Q_prime_5 = Q_prime[:decoder_token_id+1]
    K_prime_5 = K_prime[:decoder_token_id+1]
    V_prime_5 = V_prime[:decoder_token_id+1]

    QKV_dict = { 'attention_0': {"Q": Q_0, "K": K_0, "V": V_0, "Q_prime": Q_prime_0, "K_prime": K_prime_0, "V_prime": V_prime_0},
                 'attention_1': {"Q": Q_1, "K": K_1, "V": V_1, "Q_prime": Q_prime_1, "K_prime": K_prime_1, "V_prime": V_prime_1},
                 'attention_2': {"Q": Q_2, "K": K_2, "V": V_2, "Q_prime": Q_prime_2, "K_prime": K_prime_2, "V_prime": V_prime_2},
                 'attention_3': {"Q": Q_3, "K": K_3, "V": V_3, "Q_prime": Q_prime_3, "K_prime": K_prime_3, "V_prime": V_prime_3},
                 'attention_4': {"Q": Q_4, "K": K_4, "V": V_4, "Q_prime": Q_prime_4, "K_prime": K_prime_4, "V_prime": V_prime_4},
                 'attention_5': {"Q": Q_5, "K": K_5, "V": V_5, "Q_prime": Q_prime_5, "K_prime": K_prime_5, "V_prime": V_prime_5}}

    return QKV_dict 


def get_cross_atten_QKV(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id : int, N_src_tokens : int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_0 = Q[:decoder_token_id+1]
    K_0 = K[:N_src_tokens]
    V_0 = V[:N_src_tokens]
    Q_prime_0 = Q_prime[:decoder_token_id+1]
    K_prime_0 = K_prime[:decoder_token_id+1]
    V_prime_0 = V_prime[:decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_1 = Q[:decoder_token_id+1]
    K_1 = K[:N_src_tokens]
    V_1 = V[:N_src_tokens]
    Q_prime_1 = Q_prime[:decoder_token_id+1]
    K_prime_1 = K_prime[:decoder_token_id+1]
    V_prime_1 = V_prime[:decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_2 = Q[:decoder_token_id+1]
    K_2 = K[:N_src_tokens]
    V_2 = V[:N_src_tokens]
    Q_prime_2 = Q_prime[:decoder_token_id+1]
    K_prime_2 = K_prime[:decoder_token_id+1]
    V_prime_2 = V_prime[:decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_3 = Q[:decoder_token_id+1]
    K_3 = K[:N_src_tokens]
    V_3 = V[:N_src_tokens]
    Q_prime_3 = Q_prime[:decoder_token_id+1]
    K_prime_3 = K_prime[:decoder_token_id+1]
    V_prime_3 = V_prime[:decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_4 = Q[:decoder_token_id+1]
    K_4 = K[:N_src_tokens]
    V_4 = V[:N_src_tokens]
    Q_prime_4 = Q_prime[:decoder_token_id+1]
    K_prime_4 = K_prime[:decoder_token_id+1]
    V_prime_4 = V_prime[:decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q = dec_attn_input["q"].squeeze()
    K = dec_attn_input["k"].squeeze()
    V = dec_attn_input["v"].squeeze()
    Q_prime = dec_attn_input["query"].squeeze()
    K_prime = dec_attn_input["key"].squeeze()
    V_prime = dec_attn_input["value"].squeeze()
    Q_5 = Q[:decoder_token_id+1]
    K_5 = K[:N_src_tokens]
    V_5 = V[:N_src_tokens]
    Q_prime_5 = Q_prime[:decoder_token_id+1]
    K_prime_5 = K_prime[:decoder_token_id+1]
    V_prime_5 = V_prime[:decoder_token_id+1]

    QKV_dict = { 'attention_0': {"Q": Q_0, "K": K_0, "V": V_0, "Q_prime": Q_prime_0, "K_prime": K_prime_0, "V_prime": V_prime_0},
                 'attention_1': {"Q": Q_1, "K": K_1, "V": V_1, "Q_prime": Q_prime_1, "K_prime": K_prime_1, "V_prime": V_prime_1},
                 'attention_2': {"Q": Q_2, "K": K_2, "V": V_2, "Q_prime": Q_prime_2, "K_prime": K_prime_2, "V_prime": V_prime_2},
                 'attention_3': {"Q": Q_3, "K": K_3, "V": V_3, "Q_prime": Q_prime_3, "K_prime": K_prime_3, "V_prime": V_prime_3},
                 'attention_4': {"Q": Q_4, "K": K_4, "V": V_4, "Q_prime": Q_prime_4, "K_prime": K_prime_4, "V_prime": V_prime_4},
                 'attention_5': {"Q": Q_5, "K": K_5, "V": V_5, "Q_prime": Q_prime_5, "K_prime": K_prime_5, "V_prime": V_prime_5}}

    return QKV_dict 


def get_encoder_QKV(analyzer : TransformerAnalyzer, sentence_id : int, N_src_tokens : int):
    # Attention layer 0
    enc_attn_input = analyzer.enc_0_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_0 = query[:N_src_tokens]
    key_0 = key[:N_src_tokens]
    value_0 = value[:N_src_tokens]

    # Attention layer 1
    enc_attn_input = analyzer.enc_1_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_1 = query[:N_src_tokens]
    key_1 = key[:N_src_tokens]
    value_1 = value[:N_src_tokens]
    
    # Attention layer 2
    enc_attn_input = analyzer.enc_2_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_2 = query[:N_src_tokens]
    key_2 = key[:N_src_tokens]
    value_2 = value[:N_src_tokens]

    # Attention layer 3
    enc_attn_input = analyzer.enc_3_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_3 = query[:N_src_tokens]
    key_3 = key[:N_src_tokens]
    value_3 = value[:N_src_tokens]

    # Attention layer 4
    enc_attn_input = analyzer.enc_4_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_4 = query[:N_src_tokens]
    key_4 = key[:N_src_tokens]
    value_4 = value[:N_src_tokens]

    # Attention layer 5
    enc_attn_input = analyzer.enc_5_attn_probe._probe_in[sentence_id]
    query = enc_attn_input["query"].squeeze()
    key = enc_attn_input["key"].squeeze()
    value = enc_attn_input["value"].squeeze()
    query_5 = query[:N_src_tokens]
    key_5 = key[:N_src_tokens]
    value_5 = value[:N_src_tokens]

    QKV_dict = { 'attention_0': {"Q_prime": query_0, "K_prime": key_0, "V_prime": value_0},
                 'attention_1': {"Q_prime": query_1, "K_prime": key_1, "V_prime": value_1},
                 'attention_2': {"Q_prime": query_2, "K_prime": key_2, "V_prime": value_2},
                 'attention_3': {"Q_prime": query_3, "K_prime": key_3, "V_prime": value_3},
                 'attention_4': {"Q_prime": query_4, "K_prime": key_4, "V_prime": value_4},
                 'attention_5': {"Q_prime": query_5, "K_prime": key_5, "V_prime": value_5} }

    return QKV_dict 

def get_self_atten_QKV_head(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id : int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_0 = Q_head[:,:decoder_token_id+1]
    K_head_0 = K_head[:,:decoder_token_id+1]
    V_head_0 = V_head[:,:decoder_token_id+1]
    scored_V_head_0 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_1 = Q_head[:,:decoder_token_id+1]
    K_head_1 = K_head[:,:decoder_token_id+1]
    V_head_1 = V_head[:,:decoder_token_id+1]
    scored_V_head_1 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_2 = Q_head[:,:decoder_token_id+1]
    K_head_2 = K_head[:,:decoder_token_id+1]
    V_head_2 = V_head[:,:decoder_token_id+1]
    scored_V_head_2 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_3 = Q_head[:,:decoder_token_id+1]
    K_head_3 = K_head[:,:decoder_token_id+1]
    V_head_3 = V_head[:,:decoder_token_id+1]
    scored_V_head_3 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_4 = Q_head[:,:decoder_token_id+1]
    K_head_4 = K_head[:,:decoder_token_id+1]
    V_head_4 = V_head[:,:decoder_token_id+1]
    scored_V_head_4 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_5 = Q_head[:,:decoder_token_id+1]
    K_head_5 = K_head[:,:decoder_token_id+1]
    V_head_5 = V_head[:,:decoder_token_id+1]
    scored_V_head_5 = scored_V_head[:,:decoder_token_id+1]

    QKV_dict = { 'attention_0': {"Q_head": Q_head_0, "K_head": K_head_0, "V_head": V_head_0, "scored_V_head": scored_V_head_0},
                 'attention_1': {"Q_head": Q_head_1, "K_head": K_head_1, "V_head": V_head_1, "scored_V_head": scored_V_head_1},
                 'attention_2': {"Q_head": Q_head_2, "K_head": K_head_2, "V_head": V_head_2, "scored_V_head": scored_V_head_2},
                 'attention_3': {"Q_head": Q_head_3, "K_head": K_head_3, "V_head": V_head_3, "scored_V_head": scored_V_head_3},
                 'attention_4': {"Q_head": Q_head_4, "K_head": K_head_4, "V_head": V_head_4, "scored_V_head": scored_V_head_4},
                 'attention_5': {"Q_head": Q_head_5, "K_head": K_head_5, "V_head": V_head_5, "scored_V_head": scored_V_head_5}}

    return QKV_dict 

def get_cross_atten_QKV_head(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id : int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_0 = Q_head[:,:decoder_token_id+1]
    K_head_0 = K_head[:,:decoder_token_id+1]
    V_head_0 = V_head[:,:decoder_token_id+1]
    scored_V_head_0 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_1 = Q_head[:,:decoder_token_id+1]
    K_head_1 = K_head[:,:decoder_token_id+1]
    V_head_1 = V_head[:,:decoder_token_id+1]
    scored_V_head_1 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_2 = Q_head[:,:decoder_token_id+1]
    K_head_2 = K_head[:,:decoder_token_id+1]
    V_head_2 = V_head[:,:decoder_token_id+1]
    scored_V_head_2 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_3 = Q_head[:,:decoder_token_id+1]
    K_head_3 = K_head[:,:decoder_token_id+1]
    V_head_3 = V_head[:,:decoder_token_id+1]
    scored_V_head_3 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_4 = Q_head[:,:decoder_token_id+1]
    K_head_4 = K_head[:,:decoder_token_id+1]
    V_head_4 = V_head[:,:decoder_token_id+1]
    scored_V_head_4 = scored_V_head[:,:decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    Q_head = dec_attn_input["query_head"].squeeze()
    K_head = dec_attn_input["key_head"].squeeze()
    V_head = dec_attn_input["value_head"].squeeze()
    scored_V_head = dec_attn_input["scored_value"].squeeze()
    Q_head_5 = Q_head[:,:decoder_token_id+1]
    K_head_5 = K_head[:,:decoder_token_id+1]
    V_head_5 = V_head[:,:decoder_token_id+1]
    scored_V_head_5 = scored_V_head[:,:decoder_token_id+1]

    QKV_dict = { 'attention_0': {"Q_head": Q_head_0, "K_head": K_head_0, "V_head": V_head_0, "scored_V_head": scored_V_head_0},
                 'attention_1': {"Q_head": Q_head_1, "K_head": K_head_1, "V_head": V_head_1, "scored_V_head": scored_V_head_1},
                 'attention_2': {"Q_head": Q_head_2, "K_head": K_head_2, "V_head": V_head_2, "scored_V_head": scored_V_head_2},
                 'attention_3': {"Q_head": Q_head_3, "K_head": K_head_3, "V_head": V_head_3, "scored_V_head": scored_V_head_3},
                 'attention_4': {"Q_head": Q_head_4, "K_head": K_head_4, "V_head": V_head_4, "scored_V_head": scored_V_head_4},
                 'attention_5': {"Q_head": Q_head_5, "K_head": K_head_5, "V_head": V_head_5, "scored_V_head": scored_V_head_5}}

    return QKV_dict 
