# Implementation of the TransformerAnalyzer class

import numpy as np
from pathlib import Path
import os
from typing import Tuple
import torch
from tqdm import tqdm

# Definitions of the classes implemented by this application
from ProbeManager import ProbeManager


class TransformerAnalyzer:
    # Constructor
    def __init__(self, config : dict, probe_config : dict) -> None:
        super().__init__()
        # Get the model configuration parameters
        self._seq_len = config["seq_len"]
        self._d_model = config["d_model"]
        self._N_epochs = config["num_epochs"]
        self._probe_folder = config["probe_dir"]
        self._probe_config = probe_config

        # Embedding layer probes
        self._embedding_probe = ProbeManager()

        # Encoder 0 attention layer probe
        self._enc_0_attn_probe = ProbeManager()

    def run(self) -> None:
        probe_dir = Path(self._probe_folder)

        if probe_dir.exists():
            if probe_dir.is_dir() == False:
                raise ValueError(f"Invalid probe directory name: {str(probe_dir)}")
        else:
            raise RuntimeError(f"Probe directory {str(probe_dir)} does not exist")
            
        for epoch in range(self._N_epochs):
            self._embedding_probe.load(epoch, probe_dir, self._probe_config["enc_embed_layer"])
            self._enc_0_attn_probe.load(epoch, probe_dir, self._probe_config["enc_layer_0_attn"])


