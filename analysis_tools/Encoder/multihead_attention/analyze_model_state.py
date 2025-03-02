
import sys
sys.path.append('../../..')
sys.path.append('../../..utils')
import os

import torch
import torch.nn as nn
import torchinfo
import sys
import numpy as np
from pathlib import Path
from termcolor import cprint, colored
import matplotlib.pyplot as plt

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

# Class implemented by this application
from Transformer import Transformer
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_QKV import get_query_key_value_matrix
from get_sentence_tokens import get_sentence_tokens

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

def load_tokenizer(config : dict):
    # Get the model configuration parameters
    lang_src = config["lang_src"]
    lang_tgt = config["lang_tgt"]
    tokenizer_dir = config["tokenizer_dir"]

    # Load the source and target language tokenizers from the JSON file created
    # during the training of the model
    tokenizer_src_fname = Path(f"{tokenizer_dir}/tokenizer_{lang_src}.json")
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_fname))

    tokenizer_tgt_fname = Path(f"{tokenizer_dir}/tokenizer_{lang_tgt}.json")
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_fname))

    return tokenizer_src, tokenizer_tgt

def get_projection_matrices(epoch : int, model : Transformer, attention_layer : int, config : dict):
    # Get the model configuration values
    key = "model_dir"
    if key in config:
        model_folder = config["model_dir"]
    else:
        RuntimeError("Model config dictionary does not contain model_dir")
        return

    seq_len = config["seq_len"]
    d_model = config["d_model"]
    N_epochs = config["num_epochs"]

    # Directory for loading the model weights
    model_dir = Path(model_folder)
    if model_dir.exists() == False:
        RuntimeError("Model directory not found!")
        return

    # Load the model weights for the epoch
    model_filename = model_dir / f"transformer_epoch_{epoch:02d}.pt"
    print(f"Loading model file: {str(model_filename)}")
    state = torch.load(str(model_filename), weights_only=False)
    model.load_state_dict(state['model_state_dict'])

    W_q = model._encoder._layers[attention_layer]._self_attention._W_q
    W_k = model._encoder._layers[attention_layer]._self_attention._W_k
    W_v = model._encoder._layers[attention_layer]._self_attention._W_v
    W_o = model._encoder._layers[attention_layer]._self_attention._W_o

    return W_q, W_k, W_v, W_o

def projection_matrix_stats(Q, K, V, O):
    # print(f"Q shape = {Q.weight.shape}")
    # print(f"K shape = {K.weight.shape}")
    # print(f"V shape = {V.weight.shape}")
    # print(f"O shape = {O.weight.shape}")

    I = torch.eye(Q.weight.shape[0])

    Q_mat = Q(I).T
    K_mat = K(I).T
    V_mat = V(I).T
    O_mat = O(I).T

    # print(f"Q_mat shape = {Q_mat.shape}")
    # print(f"K_mat shape = {K_mat.shape}")
    # print(f"V_mat shape = {V_mat.shape}")
    # print(f"O_mat shape = {O_mat.shape}")

    Q_prime = torch.matmul(Q_mat, Q_mat)
    print(f"Rank of Q_mat = {torch.linalg.matrix_rank(Q_mat)}")

    if torch.all(torch.isclose(Q_prime, Q_mat, atol=1e-1)):
        print("Q_mat is idempotent")
    else:
        print("Q_mat is NOT idempotent")

    if torch.all(torch.isclose(Q_mat, Q_mat.T, atol=1e-1)):
        print("Q_mat is symmetric")
    else:
        print("Q_mat is NOT symmetric")

    # eigenvalues, eigenvectors = torch.linalg.eig(Q_mat) 
    # print(f"Eigenvalues = {eigenvalues}")

    K_prime = torch.matmul(K_mat, K_mat)
    print(f"Rank of K_mat = {torch.linalg.matrix_rank(K_mat)}")
    if torch.all(torch.isclose(K_prime, K_mat, atol=1e-1)):
        print("K_mat is idempotent")
    else:
        print("K_mat is NOT idempotent")

    if torch.all(torch.isclose(K_mat, K_mat.T, atol=1e-1)):
        print("K_mat is symmetric")
    else:
        print("K_mat is NOT symmetric")

    V_prime = torch.matmul(V_mat, V_mat)
    print(f"Rank of V_mat = {torch.linalg.matrix_rank(V_mat)}")
    if torch.all(torch.isclose(V_prime, V_mat, atol=1e-1)):
        print("V_mat is idempotent")
    else:
        print("V_mat is NOT idempotent")

    if torch.all(torch.isclose(V_mat, V_mat.T, atol=1e-1)):
        print("V_mat is symmetric")
    else:
        print("V_mat is NOT symmetric")

    O_prime = torch.matmul(O_mat, O_mat)
    print(f"Rank of O_mat = {torch.linalg.matrix_rank(O_mat)}")
    if torch.all(torch.isclose(O_prime, O_mat, atol=1e-1)):
        print("O_mat is idempotent")
    else:
        print("O_mat is NOT idempotent")

    if torch.all(torch.isclose(O_mat, O_mat.T, atol=1e-1)):
        print("O_mat is symmetric")
    else:
        print("O_mat is NOT symmetric")

def process_projection_matrix_MI(analyzer, W_q, W_k, W_v, W_o, QKV_list, attention_layer, sentence_id):
    QKV_dict = QKV_list[sentence_id]
    QKV_atten_dict = QKV_dict[f"attention_{attention_layer}"]
    sentence_tokens = QKV_dict["sentence_tokens"]
    x = QKV_atten_dict["x"]

    N_rows = x.shape[0]
    N_cols = x.shape[1]

    MI_estimate = np.zeros((N_rows, N_cols))
    for row in range(N_rows):
        for col in range(N_cols):
            print(f"Processing: Row = {row}/{N_rows-1}, Col = {col}/{N_cols-1}")
            q = x[row]
            w = W_q[:,col]

            # Instantiate the Mutual Information Estimator object
            # and get the probability and mutual information dictionaries
            MI_estimator = MutualInfoEstimator(q, w)

            # Get probabilities and MI estimates from the KDE estimator
            _, KDE_MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
            MI = KDE_MI_data['MI']

            # Get the MI estimate from the Kraskov estimator
            # Kraskov_MI_data = MI_estimator.kraskov_MI()
            # MI = Kraskov_MI_data['MI']

            MI_estimate[row, col] = MI

    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    fig, ax = plt.subplots()
    # img = ax.imshow(MI_estimate, cmap=plt.cm.Wistia)
    img = ax.imshow(MI_estimate.T, cmap=plt.cm.jet)
    ax.set_aspect('auto')
    ax.set_title(f"Mutual Information between q and W_q")
    ax.set_xticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=ax)
    plt.show(block=True)


# Main function of this script
def main():
    # Check if the GPU is available
    # cuda_is_avail = check_system()
    # device = torch.device("cuda:0" if cuda_is_avail else "cpu")

    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
    model_config["model_dir"] = "../../../model_data/opus_books_en_fr/weights"
    model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"
    #model_config["tokenizer_dir"] = "../../model_data_d32/opus_books_en_fr/tokens"
    #model_config["model_dir"] = "../../model_data_d32/opus_books_en_fr/weights"

    source_sequence_len = model_config["seq_len"]
    target_sequence_len = model_config["seq_len"]
    d_model = model_config["d_model"]
    # d_model = 32

    tokenizer_src, tokenizer_tgt = load_tokenizer(model_config)
    source_vocab_size = tokenizer_src.get_vocab_size()
    target_vocab_size = tokenizer_tgt.get_vocab_size()

    print(f"source: vocab_size = {source_vocab_size}")
    print(f"target: vocab_size = {target_vocab_size}")

    # Create the Transformer model
    transf_model = Transformer(source_vocab_size = source_vocab_size,
                               target_vocab_size = target_vocab_size,
                               source_sequence_len = source_sequence_len, 
                               target_sequence_len = target_sequence_len,
                               d_model = d_model)

    display_model_stats(transf_model)

    #for epoch in range(0, 20):
        #for attention_layer in range(0, 6):
            #print(f"Epoch = {epoch}, Attention Layer = {attention_layer}") 
            # get_projection_matrices(epoch, transf_model.to(device), model_config)
            #Q, K, V, O = get_projection_matrices(epoch, transf_model, attention_layer, model_config)
            #process_projection_matrices(Q, K, V, O)

    epoch = 19
    attention_layer = 5
    analyze_sentence = 0

    print(f"Epoch = {epoch}, Attention Layer = {attention_layer}")
    W_q_nn, W_k_nn, W_v_nn, W_o_nn= get_projection_matrices(epoch, transf_model, attention_layer, model_config)

    I = torch.eye(W_q_nn.weight.shape[0])
    W_q = W_q_nn(I).detach().numpy()
    W_k = W_k_nn(I).detach().numpy()
    W_v = W_v_nn(I).detach().numpy()
    W_o = W_o_nn(I).detach().numpy()

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Instantiate the Transformer's analyzer object.
    # Load the tokenizer and instantiate the ProbeManager objects
    analyzer = TransformerAnalyzer(model_config, probe_config)
    analyzer.run()

    # For this epoch, load all the encoder layer probe files from disk
    analyzer.load_encoder_probes(epoch)

    # Number of input sentences in this epoch
    N_inputs = len(analyzer.encoder_probe._probe_in)

    # This will contain the QKV dictionaries for all the attention layers
    # of all the input sentences of this epoch
    QKV_list = list()

    for sentence_id in range(N_inputs):
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

        # Get the query, key, value arrays for all the attention layers of this input sentence
        QKV_dict = get_query_key_value_matrix(analyzer, sentence_id, N_src_tokens)
        QKV_dict["sentence_tokens"] = src_sentence_tokens
        QKV_list.append(QKV_dict)

    process_projection_matrix_MI(analyzer, W_q, W_k, W_v, W_o, QKV_list, attention_layer, analyze_sentence)

# Entry point of the script
if __name__ == '__main__':
    main()