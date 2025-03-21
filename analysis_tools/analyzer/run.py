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
# from analyze_encoder_QKV import analyze_encoder_QKV
from analyze_encoder_QKV_prime import analyze_encoder_QKV_prime
from analyze_encoder_QKV_head import analyze_encoder_QKV_head

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
    print(f"Running test: {args.test}")
    
    match args.test:
        case 1:
            pass

        case 2:
            pass

        case 3:
            # Analyze Q, K, V inputs and Q', K', V' projections
            # analyze_encoder_QKV(analyzer)
            pass

        case 4:
            # Analyze Q', K', V' projections
            analyze_encoder_QKV_prime(analyzer)

        case 5:
            analyze_encoder_QKV_head(analyzer)
            pass

        case 6:
            # analyze_encoder_attention_scores(analyzer)
            pass

        case 7:
            # analyze_encoder_feedforward(analyzer)
            pass

        case _:
            print("Invalid test id")
            return

    # print("Press any key to exit the program ...")
    # input()


# Entry point of the program
if __name__ == '__main__':
    main()
