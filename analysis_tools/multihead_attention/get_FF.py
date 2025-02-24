
import sys
sys.path.append('../..')
sys.path.append('../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_FF_input_output(analyzer : TransformerAnalyzer, sentence_id : int, N_src_tokens : int):
    # Encoder layer 0
    enc_ff_input = analyzer.enc_0_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_0_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_0 = ff_in_full[:N_src_tokens]
    ff_out_0 = ff_out_full[:N_src_tokens]

    # Encoder layer 1
    enc_ff_input = analyzer.enc_1_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_1_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_1 = ff_in_full[:N_src_tokens]
    ff_out_1 = ff_out_full[:N_src_tokens]

    # Encoder layer 2
    enc_ff_input = analyzer.enc_2_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_2_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_2 = ff_in_full[:N_src_tokens]
    ff_out_2 = ff_out_full[:N_src_tokens]

    # Encoder layer 3
    enc_ff_input = analyzer.enc_3_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_3_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_3 = ff_in_full[:N_src_tokens]
    ff_out_3 = ff_out_full[:N_src_tokens]

    # Encoder layer 4
    enc_ff_input = analyzer.enc_4_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_4_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_4 = ff_in_full[:N_src_tokens]
    ff_out_4 = ff_out_full[:N_src_tokens]

    # Encoder layer 5
    enc_ff_input = analyzer.enc_5_feedforward_probe._probe_in[sentence_id]
    enc_ff_output = analyzer.enc_5_feedforward_probe._probe_out[sentence_id]
    ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
    ff_in_5 = ff_in_full[:N_src_tokens]
    ff_out_5 = ff_out_full[:N_src_tokens]

    ff_dict = { 'ff_0': {"ff_in": ff_in_0, "ff_out": ff_out_0},
                'ff_1': {"ff_in": ff_in_1, "ff_out": ff_out_1},
                'ff_2': {"ff_in": ff_in_2, "ff_out": ff_out_2},
                'ff_3': {"ff_in": ff_in_3, "ff_out": ff_out_3},
                'ff_4': {"ff_in": ff_in_4, "ff_out": ff_out_4},
                'ff_5': {"ff_in": ff_in_5, "ff_out": ff_out_5}}

    return ff_dict 
