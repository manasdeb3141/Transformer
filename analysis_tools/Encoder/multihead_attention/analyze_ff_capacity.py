
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time


# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_FF import get_FF_input_output
from BlahutArimoto import blahut_arimoto_capacity


def stack_FF_matrix(FF_list, N_inputs):
    ff_in_0_array = None; ff_out_0_array = None
    ff_in_1_array = None; ff_out_1_array = None
    ff_in_2_array = None; ff_out_2_array = None
    ff_in_3_array = None; ff_out_3_array = None
    ff_in_4_array = None; ff_out_4_array = None
    ff_in_5_array = None; ff_out_5_array = None

    for i in range(N_inputs):
        ff_in_0 = FF_list[i]['ff_0']['ff_in']
        ff_out_0 = FF_list[i]['ff_0']['ff_out']
        ff_in_1 = FF_list[i]['ff_1']['ff_in']
        ff_out_1 = FF_list[i]['ff_1']['ff_out']
        ff_in_2 = FF_list[i]['ff_2']['ff_in']
        ff_out_2 = FF_list[i]['ff_2']['ff_out']
        ff_in_3 = FF_list[i]['ff_3']['ff_in']
        ff_out_3 = FF_list[i]['ff_3']['ff_out']
        ff_in_4 = FF_list[i]['ff_4']['ff_in']
        ff_out_4 = FF_list[i]['ff_4']['ff_out']
        ff_in_5 = FF_list[i]['ff_5']['ff_in']
        ff_out_5 = FF_list[i]['ff_5']['ff_out']

        if ff_in_0_array is None:
            ff_in_0_array = ff_in_0
            ff_out_0_array = ff_out_0
        else:
            ff_in_0_array = np.vstack((ff_in_0_array, ff_in_0))
            ff_out_0_array = np.vstack((ff_out_0_array, ff_out_0))

        if ff_in_1_array is None:
            ff_in_1_array = ff_in_1
            ff_out_1_array = ff_out_1
        else:
            ff_in_1_array = np.vstack((ff_in_1_array, ff_in_1))
            ff_out_1_array = np.vstack((ff_out_1_array, ff_out_1))

        if ff_in_2_array is None:
            ff_in_2_array = ff_in_2
            ff_out_2_array = ff_out_2
        else:
            ff_in_2_array = np.vstack((ff_in_2_array, ff_in_2))
            ff_out_2_array = np.vstack((ff_out_2_array, ff_out_2))

        if ff_in_3_array is None:
            ff_in_3_array = ff_in_3
            ff_out_3_array = ff_out_3
        else:
            ff_in_3_array = np.vstack((ff_in_3_array, ff_in_3))
            ff_out_3_array = np.vstack((ff_out_3_array, ff_out_3))

        if ff_in_4_array is None:
            ff_in_4_array = ff_in_4
            ff_out_4_array = ff_out_4
        else:
            ff_in_4_array = np.vstack((ff_in_4_array, ff_in_4))
            ff_out_4_array = np.vstack((ff_out_4_array, ff_out_4))

        if ff_in_5_array is None:
            ff_in_5_array = ff_in_5
            ff_out_5_array = ff_out_5
        else:
            ff_in_5_array = np.vstack((ff_in_5_array, ff_in_5))
            ff_out_5_array = np.vstack((ff_out_5_array, ff_out_5))

    ff_dict = { 'ff_0': {"ff_in": ff_in_0_array, "ff_out": ff_out_0_array},
                'ff_1': {"ff_in": ff_in_1_array, "ff_out": ff_out_1_array},
                'ff_2': {"ff_in": ff_in_2_array, "ff_out": ff_out_2_array},
                'ff_3': {"ff_in": ff_in_3_array, "ff_out": ff_out_3_array},
                'ff_4': {"ff_in": ff_in_4_array, "ff_out": ff_out_4_array},
                'ff_5': {"ff_in": ff_in_5_array, "ff_out": ff_out_5_array}}

    return ff_dict 

def compute_FF_capacity(FF_inout, encoder_layer, prob_dict=None):
    if prob_dict is None:
        ff_dict = FF_inout[f"ff_{encoder_layer}"]
        ff_in = ff_dict["ff_in"]
        ff_out = ff_dict["ff_out"]

        X = ff_in.reshape(-1, 1)
        Y = ff_out.reshape(-1, 1)

        # Compute joint PDF between FF input and output
        print("Computing the joint PDF between the FF input and output ...")
        start_time = time.time()
        MI_estimator = MutualInfoEstimator(X, Y)
        prob_dict, mi_dict = MI_estimator.kernel_MI(KDE_module='sklearn', N_points=100)
        print(f"Time taken: {time.time() - start_time}")

    # Compute capacity
    P_XY = prob_dict["P_XY"]
    P_X = prob_dict["P_X"]
    P_Y = prob_dict["P_Y"]
    P_Y_given_X = P_XY / P_X
    row_sum = P_Y_given_X.sum(axis=1, keepdims=True)
    P_Y_given_X = P_Y_given_X / row_sum
    C_FF, _ = blahut_arimoto_capacity(P_Y_given_X)

    return C_FF, prob_dict

# Main function of this script
def main():
    print("Computing the capcity of the feed-forward layer of the encoder ...")

    epoch = 19
    encoder_layer = 2
    sentence_id = 3

    save_file = Path(f"data/ff_capacity_epoch_{epoch}_layer_{encoder_layer}_sentence_{sentence_id}.pt")

    if save_file.exists():
        print(f"Feed-forward capacity data file {str(save_file)} found. Loading it ...")
        prob_data = torch.load(save_file, weights_only=False)
        FF_capacity, _ = compute_FF_capacity(None, None, prob_data)
    else:
        cfg_obj = LangModelConfig()
        model_config = cfg_obj.get_config()

        model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
        model_config["model_dir"] = "../../../model_data/opus_books_en_fr/weights"
        model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"
        #model_config["tokenizer_dir"] = "../../../model_data_d32/opus_books_en_fr/tokens"
        #model_config["model_dir"] = "../../../model_data_d32/opus_books_en_fr/weights"

        # Dictionary of probe file names
        probe_config = cfg_obj.get_probes()

        # Instantiate the Transformer's analyzer object.
        # Load the tokenizer and instantiate the ProbeManager objects
        analyzer = TransformerAnalyzer(model_config, probe_config)
        analyzer.run()

        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)

        # This list contains all the FF input and output arrays for all the attention layers of all the input sentences
        FF_list = list()
        
        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        for sentence_id in range(N_inputs):
            # Get the tokens of the source and target sentences
            N_src_tokens, _, _, _ = get_sentence_tokens(analyzer, sentence_id)

            # Get the query, key, value arrays for all the attention layers of this input sentence
            FF_inout = get_FF_input_output(analyzer, sentence_id, N_src_tokens)
            FF_list.append(FF_inout)

        # Concatenate all the FF input and output arrays
        FF_stacked = stack_FF_matrix(FF_list, N_inputs)

        FF_capacity, prob_data = compute_FF_capacity(FF_stacked, encoder_layer)

        # Save the file
        torch.save(prob_data, save_file)
        
    print(f"Feed-forward capacity: {FF_capacity}")
    

# Entry point of the script
if __name__ == '__main__':
    main()