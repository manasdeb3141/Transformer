
import sys
sys.path.append('../..')
sys.path.append('../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_FF_input_output(analyzer : TransformerAnalyzer, sentence_id : int, token_id : int):
    # Encoder layer 0
    dec_ff_input = analyzer.dec_0_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_0_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_0 = ff_in_full[:token_id+1]
    ff_out_0 = ff_out_full[:token_id+1]

    # Encoder layer 1
    dec_ff_input = analyzer.dec_1_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_1_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_1 = ff_in_full[:token_id+1]
    ff_out_1 = ff_out_full[:token_id+1]

    # Encoder layer 2
    dec_ff_input = analyzer.dec_2_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_2_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_2 = ff_in_full[:token_id+1]
    ff_out_2 = ff_out_full[:token_id+1]

    # Encoder layer 3
    dec_ff_input = analyzer.dec_3_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_3_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_3 = ff_in_full[:token_id+1]
    ff_out_3 = ff_out_full[:token_id+1]

    # Encoder layer 4
    dec_ff_input = analyzer.dec_4_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_4_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_4 = ff_in_full[:token_id+1]
    ff_out_4 = ff_out_full[:token_id+1]

    # Encoder layer 5
    dec_ff_input = analyzer.dec_5_feedforward_probe._probe_in[sentence_id][token_id]
    dec_ff_output = analyzer.dec_5_feedforward_probe._probe_out[sentence_id][token_id]
    ff_in_full = dec_ff_input.squeeze()     # seq_len x d_model
    ff_out_full = dec_ff_output.squeeze()   # seq_len x d_model
    ff_in_5 = ff_in_full[:token_id+1]
    ff_out_5 = ff_out_full[:token_id+1]

    ff_dict = { 'ff_0': {"ff_in": ff_in_0, "ff_out": ff_out_0},
                'ff_1': {"ff_in": ff_in_1, "ff_out": ff_out_1},
                'ff_2': {"ff_in": ff_in_2, "ff_out": ff_out_2},
                'ff_3': {"ff_in": ff_in_3, "ff_out": ff_out_3},
                'ff_4': {"ff_in": ff_in_4, "ff_out": ff_out_4},
                'ff_5': {"ff_in": ff_in_5, "ff_out": ff_out_5}}

    return ff_dict 
