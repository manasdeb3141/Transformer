# Implementation of the LanguageTranslator class

from pathlib import Path
import os
import torch

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer

# Definitions of the classes implemented by this application
from Transformer import Transformer

class LanguageTranslator:
    # Constructor
    def __init__(self, device : torch.device) -> None:
        super().__init__()
        self._device = device
        self._tokenizer_src = None
        self._tokenizer_tgt = None
        self._seq_len = 0
        self._model = None

    # Generate a causal mask
    def __causal_mask(self, size : int) -> torch.Tensor:
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    def load_tokenizer(self, config : dict):
        # Get the model configuration parameters
        lang_src = config["lang_src"]
        lang_tgt = config["lang_tgt"]
        tokenizer_dir = config["tokenizer_dir"]

        # Load the source and target language tokenizers from the JSON file created
        # during the training of the model
        tokenizer_src_fname = Path(f"{tokenizer_dir}/tokenizer_{lang_src}.json")
        self._tokenizer_src = Tokenizer.from_file(str(tokenizer_src_fname))

        tokenizer_tgt_fname = Path(f"{tokenizer_dir}/tokenizer_{lang_tgt}.json")
        self._tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_fname))

    def get_vocab_size(self, selection : str) -> int:
        if selection == "src":
            return self._tokenizer_src.get_vocab_size()
        elif selection == "tgt":
            return self._tokenizer_tgt.get_vocab_size()
        else:
            raise ValueError(f"Invalid parameter {selection} passed to LanguageTranslator.get_vocab_str")
            return 0

    def load_model_weights(self, config : dict, model : Transformer) -> bool:
        model_folder = config["model_dir"]
        preload_file = config['preload']
        self._seq_len = config['seq_len']
        self._model = model

        # Directory containing the model weights saved during training
        model_dir = Path(model_folder)

        if preload_file == 'latest':
            # Find the model weights file saved after the last epoch of training
            model_files = [f for f in os.listdir(model_dir) if (os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".pt"))]
            if len(model_files) == 0:
                return False

            # Sort the files in ascending order. The last file in the
            # sorted list is the latest model weights file that was saved
            model_files.sort()
            latest_model_file = model_files[-1]
            preload_file_path = model_dir / latest_model_file
        else:
            # Caller specified model filename
            preload_file_path = model_dir / preload_file
            if os.path.isfile(preload_file_path) is False:
                return False

        # Load the model weights
        print(f"Preloading model file: {str(preload_file_path)}")
        state = torch.load(str(preload_file_path))
        self._model.load_state_dict(state['model_state_dict'])

        return True

    def translate(self, sentence : str) -> str:
        if (self._model == None):
            print("ERROR: Transformer model not loaded!")
            return None

        self._model.eval()
        with torch.no_grad():
            # Compute the encoder output and use it for every generation step of the decoder
            source = self._tokenizer_src.encode(sentence)
            source = torch.cat([
                torch.tensor([self._tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([self._tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([self._tokenizer_src.token_to_id('[PAD]')] * (self._seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(self._device)
            source_mask = (source != self._tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(self._device)
            encoder_output = self._model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(self._tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(self._device)

        while decoder_input.size(1) < self._seq_len:
            # Create the decoder mask and get the decoder output
            temp_mask = self.__causal_mask(decoder_input.size(1))
            decoder_mask = temp_mask.type_as(source_mask).to(self._device)
            decoder_output = self._model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # Project the next token
            prob = self._model.project(decoder_output[:,-1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self._device)], dim=1)

            # print the translated word
            translated_word = self._tokenizer_tgt.decode([next_word.item()])
            print(f"{translated_word}", end=' ')

            # break if we predict the end of sentence token
            if next_word == self._tokenizer_tgt.token_to_id('[EOS]'):
                break
            
        return self._tokenizer_tgt.decode(decoder_input[0].tolist())


