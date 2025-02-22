
import sys
sys.path.append('../..')
sys.path.append('../utils')

import torch
import torch.nn as nn
import os
import argparse
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

# Functions implemented by this application
from process_QKV_matrix import process_QKV_matrix
from process_QKV_heads import process_QKV_heads
from process_attention_scores import process_attention_scores
from process_mi_attention_scores import process_mi_attention_scores


def process_probes(test_id, analyzer):
    print(f"Running test: {test_id}")
    
    match test_id:
        case 1:
            # Compute the column entropy and mutual information of the Query, Key and Value arrays
            # of the multihead attention layers
            process_QKV_matrix(analyzer)

        case 2:
            # Compute the column entropy and mutual information of the Query, Key and Value heads
            # of the multihead attention layers
            process_QKV_heads(analyzer)

        case 3:
            process_attention_scores(analyzer)

        case 4:
            process_mi_attention_scores(analyzer)

        case _:
            print("Invalid test id")
            return

    print("Press any key to exit the program ...")
    input()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs='?', help="Test id to run", type=int)

    # Parse the argument
    args = parser.parse_args()

    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../../model_data/opus_books_en_fr/probes_8"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Instantiate the Transformer's analyzer object.
    # Load the tokenizer and instantiate the ProbeManager objects
    analyzer = TransformerAnalyzer(model_config, probe_config)
    analyzer.run()

    # Run the requested test
    process_probes(args.test, analyzer)


# Entry point of the program
if __name__ == '__main__':
    main()