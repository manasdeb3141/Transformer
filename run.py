import torch
import torch.nn as nn
import torchinfo
import os
import sys
from pathlib import Path
import argparse
from termcolor import cprint, colored
import signal

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Class implemented by this application
from Transformer import Transformer
from TrainingDataset import TrainingDataset
from ModelConfig import LangModelConfig
from ModelTrainer import ModelTrainer
from LanguageTranslator import LanguageTranslator
from TransformerProbe import TransformerProbe
from TransformerAnalyzer import TransformerAnalyzer

def display_model_stats(model : Transformer):
    print("\n")
    print(colored('Transformer model parameters:', 'green', attrs=['bold', 'underline']))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    print("\n")
    print(colored('Transformer model summary:', 'green', attrs=['bold', 'underline']))
    for layer_name, params in model.named_parameters():
        print(layer_name, params.shape)
    print('\n'*2)

    torchinfo.summary(model)
    print('\n'*2)

# Function to check the number of CPU cores and GPUs
def check_system() -> bool:
    print(f"Pytorch version = {torch.__version__}")

    N_CPU = os.cpu_count()
    print(f"Number of CPUs available: {N_CPU}")

    cuda_is_avail =  torch.cuda.is_available()

    if (cuda_is_avail):
        print('Pytorch is using one or more GPUs with CUDA')

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs is: {num_gpus}")

        print(colored('GPU device list:', 'green', attrs=['bold', 'underline']))
        for i in range(num_gpus):
            device_prop = torch.cuda.get_device_properties(i)
            print('    Device Name =', end=' ')
            cprint(f"{device_prop.name}", 'red', attrs=['bold'])
            print('    Total memory =', end=' ')
            cprint(f"{(device_prop.total_memory / (1024 ** 3)):.2f} GB ", 'red', attrs=['bold'])
            print('    Multiprocessor count =', end=' ')
            cprint(f"{device_prop.multi_processor_count}", 'red', attrs=['bold'])
            print('    Major version = ', colored(f"{device_prop.major}", 'red', attrs=['bold']), ' Minor Version = ', colored(f"{device_prop.minor}", 'red', attrs=['bold']))
            print('\n')
    else:
        print('Pytorch is NOT using a GPU')

    return cuda_is_avail


# Main function of this script
def main():
    usage_msg = "Please specify a valid mode to run the Transformer model"

    # Support command line arguments to train and translate
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the Transformer model", action="store_true")
    parser.add_argument("--trans", help="Run the Tranformer model as a language translator", action="store_true")
    parser.add_argument("--probe", help="Probe the layers of each epoc of the Tranformer model with the validation data", action="store_true")
    parser.add_argument("--analyze", help="Analyze the Transformer model probes", action="store_true")

    # Parse the argument
    args = parser.parse_args()

    # Check if the GPU is available
    cuda_is_avail = check_system()
    device = torch.device("cuda:0" if cuda_is_avail else "cpu")

    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    if args.train:   
        #
        # Train the Transformer model
        #

        # Default hyperparameters of the Transformer model
        source_vocab_size = 1000
        target_vocab_size = 1000
        source_sequence_len = 100
        target_sequence_len = 100
        d_model = 512

        # Get the source and target language datset and tokens
        train_ds = TrainingDataset()
        ds_dict = train_ds.get_language_dataset(model_config)

        max_seq_len_src = ds_dict["max_len_src"]
        max_seq_len_tgt = ds_dict["max_len_tgt"]

        # Get the source and target vocabulary size
        # Set the other Transformer model parameters
        # according to the configuration dictionary
        source_vocab_size = train_ds.src_vocab_size()
        target_vocab_size = train_ds.tgt_vocab_size()
        source_sequence_len = model_config["seq_len"]
        target_sequence_len = model_config["seq_len"]
        d_model = model_config["d_model"]

        print(f"source: vocab_size = {source_vocab_size}, max_sequence_len = {max_seq_len_src}")
        print(f"target: vocab_size = {target_vocab_size}, max_sequence_len = {max_seq_len_tgt}")

        # Create the Transformer model
        transf_model = Transformer(source_vocab_size = source_vocab_size, 
                                target_vocab_size = target_vocab_size,
                                source_sequence_len = source_sequence_len, 
                                target_sequence_len = target_sequence_len,
                                d_model = d_model)

        display_model_stats(transf_model)

        # Train the transformer model as a language translator
        transf_trainer = ModelTrainer(device, transf_model.to(device))
        transf_trainer.train(model_config, ds_dict)
    elif args.trans:
        #
        # Run the Transformer model as a translator
        #

        source_sequence_len = model_config["seq_len"]
        target_sequence_len = model_config["seq_len"]
        d_model = model_config["d_model"]

        # Create the translator object and load the tokenizer 
        lang_translator = LanguageTranslator(device)
        lang_translator.load_tokenizer(model_config)

        source_vocab_size = lang_translator.get_vocab_size("src")
        target_vocab_size = lang_translator.get_vocab_size("tgt")

        print(f"source: vocab_size = {source_vocab_size}")
        print(f"target: vocab_size = {target_vocab_size}")

        # Create the Transformer model
        transf_model = Transformer(source_vocab_size = source_vocab_size,
                                target_vocab_size = target_vocab_size,
                                source_sequence_len = source_sequence_len, 
                                target_sequence_len = target_sequence_len,
                                d_model = d_model)

        display_model_stats(transf_model)

        # Load the trained model weights
        if lang_translator.load_model_weights(model_config, transf_model.to(device)) == False:
            print("ERROR loading the Transformer model!")
            return

        # Continuously prompts the user for input text and translates
        # and prints the translated text until the user hits CTRL+C
        lang_translator.run()
    elif args.probe:
        #
        # Probe the layers of the Transformer with the validation data
        #
        print("Probing the Transformer layers with the validation dataset ...")

        source_sequence_len = model_config["seq_len"]
        target_sequence_len = model_config["seq_len"]
        d_model = model_config["d_model"]

        # Dictionary of probe file names
        model_probes = cfg_obj.get_probes()

        # Create the probe object and load the tokenizer 
        probe = TransformerProbe(device)
        probe.load_tokenizer(model_config)

        source_vocab_size = probe.get_vocab_size("src")
        target_vocab_size = probe.get_vocab_size("tgt")

        print(f"source: vocab_size = {source_vocab_size}")
        print(f"target: vocab_size = {target_vocab_size}")

        # Create the Transformer model
        transf_model = Transformer(source_vocab_size = source_vocab_size,
                                target_vocab_size = target_vocab_size,
                                source_sequence_len = source_sequence_len, 
                                target_sequence_len = target_sequence_len,
                                d_model = d_model)

        display_model_stats(transf_model)

        # Loads the model weights for the different epochs, runs the validation dataset
        # through each load and saves the salient Tensors to disk for offline analysis
        probe.run(transf_model.to(device), model_config, model_probes)
    elif args.analyze:
        # Dictionary of probe file names
        model_probes = cfg_obj.get_probes()

        # Analyze the Transformer probes
        analyzer = TransformerAnalyzer(model_config, model_probes)
        analyzer.run()
    else:
        print(usage_msg)
        parser.print_help()
        return


# Entry point of the script
if __name__ == '__main__':
    main()