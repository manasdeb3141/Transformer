# Implementation of the TransformerProbe class

import numpy as np
from pathlib import Path
import os
from typing import Tuple
import shutil
import torch
import torchmetrics.text
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer

# Definitions of the classes implemented by this application
from Transformer import Transformer
from MultiheadAttention import MultiheadAttention
from BilingualDataset import BilingualDataset
from ProbeManager import ProbeManager

test_vocab = [
    { 
        "id" : 1,
        "translation": 
        {
            "en" : "The dog ate the food because it was delicious",
            "fr" : "Le chien a mangé la nourriture parce que c'était délicieux" 
        }
    },
    {
        "id" : 2,
        "translation" : 
        {
            "en" : "The dog ate the food because it was hungry",
            "fr" : "Le chien a mangé la nourriture parce qu'il avait faim"
        }
    }
]
    

class TransformerProbe:
    # Constants
    MAX_PROBE_INPUT_DATA = 100
    
    # Constructor
    def __init__(self, device : torch.device) -> None:
        super().__init__()
        self._device = device
        self._epoch = 0
        self._input_count = 0
        self._tokenizer_src = None
        self._tokenizer_tgt = None
        self._seq_len = 0
        self._d_model = 0
        self._model = None
        self._val_dataloader = None
        self._global_step = 0
        self._writer = None

        # Performance metrics
        self._cer_list = list()
        self._wer_list = list()
        self._bleu_list = list()

        self._enc_embedding_probe = None        # Encoder's embedding layer probe
        self._enc_0_attn_probe = None           # Encoder 0 attention layer probe
        self._enc_0_feedforward_probe = None    # Encoder 0 feedforward layer probe
        self._enc_5_attn_probe = None           # Encoder 5 attention layer probe
        self._enc_5_feedforward_probe = None    # Encoder 5 feedforward layer probe
        self._encoder_probe = None              # Encoder block's input and output probe

        self._dec_embedding_probe = None        # Decoder's embedding layer probe
        self._dec_0_attn_probe = None           # Decoder 0 attention layer probe
        self._dec_0_cross_attn_probe = None     # Decoder 0 cross-attention layer probe
        self._dec_0_feedforward_probe = None    # Decoder 0 feedforward layer probe
        self._dec_5_attn_probe = None           # Decoder 5 attention layer probe
        self._dec_5_cross_attn_probe = None     # Decoder 5 cross-attention layer probe
        self._dec_5_feedforward_probe = None    # Decoder 5 feedforward layer probe
        self._decoder_probe = None              # Decoder block's input and output probe

        self._projection_probe = None           # Projection layer's input and output probe


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
            raise ValueError(f"Invalid parameter {selection} passed to TransformerProbe.get_vocab_str")

        return None 

    def run(self, model : Transformer, config : dict, probes : dict) -> None:
        # Get the model configuration values
        key = "model_dir"
        if key in config:
            model_folder = config["model_dir"]
        else:
            RuntimeError("TransformerProbe.run(): Model config dictionary does not contain model_dir")
            return

        key = "probe_dir"
        if key in config:
            probe_folder = config[key]
        else:
            RuntimeError("TransformerProbe.run(): Model config dictionary does not contain probe_dir")
            return

        key = "dataset_dir"
        if key in config:
            dataset_folder = config[key]
        else:
            RuntimeError("TransformerProbe.run(): Model config dictionary does not contain dataset_dir")
            return
        
        lang_src = config["lang_src"]
        lang_tgt = config["lang_tgt"]
        self._seq_len = config["seq_len"]
        self._d_model = config["d_model"]
        N_epochs = config["num_epochs"]

        # Directory for loading the model weights
        model_dir = Path(model_folder)
        if model_dir.exists() == False:
            RuntimeError("TransformerProbe.run(): Model directory not found!")
            return

        # Save the model to a member variable so that it can be
        # accessed by other methods of this class
        self._model = model

        #
        # Hook the specific layers of the model
        #
        #
        # Encoder hooks
        #
        enc_embed_hook_handle = self._model._source_embed.register_forward_hook(self.enc_embedding_hook)
        enc0_attn_hook_handle = self._model._encoder._layers[0]._self_attention.register_forward_hook(self.enc0_attention_hook)
        enc0_feedforward_hook_handle = self._model._encoder._layers[0]._feed_forward.register_forward_hook(self.enc0_feedforward_hook)
        enc5_attn_hook_handle = self._model._encoder._layers[5]._self_attention.register_forward_hook(self.enc5_attention_hook)
        enc5_feedforward_hook_handle = self._model._encoder._layers[5]._feed_forward.register_forward_hook(self.enc5_feedforward_hook)
        encoder_hook_handle = self._model._encoder.register_forward_hook(self.encoder_hook)

        # Decoder hooks
        dec_embed_hook_handle = self._model._target_embed.register_forward_hook(self.dec_embedding_hook)
        dec0_attn_hook_handle = self._model._decoder._layers[0]._self_attention.register_forward_hook(self.dec0_attention_hook)
        dec0_cross_attn_hook_handle = self._model._decoder._layers[0]._cross_attention.register_forward_hook(self.dec0_cross_attention_hook)
        dec0_feedforward_hook_handle = self._model._decoder._layers[0]._feed_forward.register_forward_hook(self.dec0_feedforward_hook)
        dec5_attn_hook_handle = self._model._decoder._layers[5]._self_attention.register_forward_hook(self.dec5_attention_hook)
        dec5_cross_attn_hook_handle = self._model._decoder._layers[5]._cross_attention.register_forward_hook(self.dec5_cross_attention_hook)
        dec5_feedforward_hook_handle = self._model._decoder._layers[5]._feed_forward.register_forward_hook(self.dec5_feedforward_hook)
        decoder_hook_handle = self._model._decoder.register_forward_hook(self.decoder_hook)

        # Projection layer hook
        projection_hook_handle = self._model._projection.register_forward_hook(self.projection_hook)

        # Load the model validation raw dataset
        dataset_dir = Path(dataset_folder)
        dataset_fname = dataset_dir / "validation_data.pt"
        val_ds_raw = torch.load(dataset_fname)

        # Create the Bilingual dataset from the raw dataset (batch size is 1 for the validation phase)
        # val_ds = BilingualDataset(val_ds_raw, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, self._seq_len)
        val_ds = BilingualDataset(test_vocab, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, self._seq_len)
        self._val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        # If it exists remove it and create it fresh
        probe_dir = Path(probe_folder)
        if probe_dir.exists():
            if probe_dir.is_dir() == False:
                raise ValueError(f"Invalid probe directory name: {str(probe_dir)}")

            # Delete all the existing sub-directories in the probe directory
            for subdir_path in probe_dir.iterdir():
                if subdir_path.is_dir():
                    shutil.rmtree(subdir_path)
        else:
            # Create the directory for storing the model probes
            probe_dir.mkdir(parents=True)

        # Override N_epochs for testing
        # N_epochs = 2

        # Instantiate the probe classes which will in turn initialize the 
        # memory for storing the probes
        N_inputs = min(self.MAX_PROBE_INPUT_DATA, len(self._val_dataloader))

        # Encoder probe objects
        self._enc_embedding_probe = ProbeManager(N_inputs)
        self._enc_0_attn_probe = ProbeManager(N_inputs)
        self._enc_0_feedforward_probe = ProbeManager(N_inputs)
        self._enc_5_attn_probe = ProbeManager(N_inputs)
        self._enc_5_feedforward_probe = ProbeManager(N_inputs)
        self._encoder_probe = ProbeManager(N_inputs)

        # Decoder probe objects
        self._dec_embedding_probe = ProbeManager(N_inputs)
        self._dec_0_attn_probe = ProbeManager(N_inputs)
        self._dec_0_cross_attn_probe = ProbeManager(N_inputs)
        self._dec_0_feedforward_probe = ProbeManager(N_inputs)
        self._dec_5_attn_probe = ProbeManager(N_inputs)
        self._dec_5_cross_attn_probe = ProbeManager(N_inputs)
        self._dec_5_feedforward_probe = ProbeManager(N_inputs)
        self._decoder_probe = ProbeManager(N_inputs)

        # Projection layer probe object
        self._projection_probe = ProbeManager(N_inputs, True)

        # Start the probing
        for epoch in range(N_epochs):
            torch.cuda.empty_cache()

            # Save the epoch
            self._epoch = epoch

            # Load the model weights for the epoch
            model_filename = model_dir / f"transformer_epoch_{epoch:02d}.pt"
            print(f"Loading model file: {str(model_filename)}")
            state = torch.load(str(model_filename))
            self._model.load_state_dict(state['model_state_dict'])

            # Total number of training steps across all epochs
            self._global_step = state['global_step']

            # Run the validation dataset on the model
            self.__validate(epoch)

            # Save the current epoch's probes
            # self._enc_embedding_probe.save(epoch, probe_dir, probes["enc_embed_layer"])
            # self._enc_0_attn_probe.save(epoch, probe_dir, probes["enc_layer_0_attn"])
            # self._enc_0_feedforward_probe.save(epoch, probe_dir, probes["enc_layer_0_feedforward"])
            # self._enc_5_attn_probe.save(epoch, probe_dir, probes["enc_layer_5_attn"])
            # self._enc_5_feedforward_probe.save(epoch, probe_dir, probes["enc_layer_5_feedforward"])
            # self._encoder_probe.save(epoch, probe_dir, probes["encoder_block_probe"])
            
        # Encoder hook cleanup
        enc_embed_hook_handle.remove()
        enc0_attn_hook_handle.remove()
        enc0_feedforward_hook_handle.remove() 
        enc5_attn_hook_handle.remove()
        enc5_feedforward_hook_handle.remove() 
        encoder_hook_handle.remove() 

        # Decoder hook cleanup
        dec_embed_hook_handle.remove()
        dec0_attn_hook_handle.remove()
        dec0_cross_attn_hook_handle.remove()
        dec0_feedforward_hook_handle.remove()
        dec5_attn_hook_handle.remove()
        dec5_cross_attn_hook_handle.remove()
        dec5_feedforward_hook_handle.remove()
        decoder_hook_handle.remove()

        # Projection hook cleanup
        projection_hook_handle.remove()


    # Generate a causal mask
    def __causal_mask(self, size : int) -> torch.Tensor:
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    # Method to decode using a single encoder output
    def __greedy_decode(self, source, source_mask):
        sos_idx = self._tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = self._tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = self._model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(self._device)

        while True:
            if decoder_input.size(1) == self._seq_len:
                break

            # build mask for target
            temp_mask = self.__causal_mask(decoder_input.size(1))
            decoder_mask = temp_mask.type_as(source_mask).to(self._device)

            # calculate output
            out = self._model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = self._model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self._device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)         

    # Model training validation method
    def __validate(self, epoch) -> None:
        source_texts = list()
        expected = list()
        predicted = list()

        self._model.eval()
        max_validn_count = min(self.MAX_PROBE_INPUT_DATA, len(self._val_dataloader))

        with torch.no_grad():
            self._input_count = 0

            # Progress bar
            batch_iterator = tqdm(self._val_dataloader, desc=f"Processing Epoch {epoch:02d}")

            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(self._device) # (batch_len, seq_len)
                encoder_mask = batch['encoder_mask'].to(self._device) # (batch_len, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = self.__greedy_decode(encoder_input, encoder_mask)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = self._tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                
                # Print the source, target and model output
                if self._input_count >= (max_validn_count-2):
                    batch_iterator.write('-'*80)
                    batch_iterator.write(f"{f'SOURCE: ':>12}{source_text}")
                    batch_iterator.write(f"{f'TARGET: ':>12}{target_text}")
                    batch_iterator.write(f"{f'PREDICTED: ':>12}{model_out_text}\n")

                if self._input_count == max_validn_count:
                    batch_iterator.write('-'*80)
                    break

                self._input_count += 1

        if self._writer:
            # Compute the char error rate 
            metric = torchmetrics.text.CharErrorRate()
            cer = metric(predicted, expected)
            self._cer_list.append(cer)

            # Compute the word error rate
            metric = torchmetrics.text.WordErrorRate()
            wer = metric(predicted, expected)
            self._wer_list.append(cer)

            # Compute the BLEU metric
            metric = torchmetrics.text.BLEUScore()
            bleu = metric(predicted, expected)
            self._bleu_list.append(cer)

    #
    # ------------------------------------ Hooks into the Transformer layers --------------------------------------------
    #
    def __process_embedding_hook(self, module, input, output) -> Tuple[np.ndarray, np.ndarray]:
        in_val = input[0].detach().cpu().numpy()        # shape = (1, seq_len)
        out_val = output.detach().cpu().numpy()         # shape = (1, seq_len, d_model)

        return in_val, out_val

    def __process_attention_hook(self, module, input, output) -> Tuple[dict, np.ndarray]:
        # q, k, v have the following shape:
        # (1, seq_len, d_model) => (1, 350, 512)
        q = input[0]
        k = input[1]
        v = input[2]

        # This has shape (1, 1, 1, seq_len) => (1, 1, 1, 350)
        mask = input[3]

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        query = module._W_q(q)

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        key = module._W_k(k)

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        value = module._W_v(v)

        # Split the query, key, value matrices into h parts along the embedding dimension

        # (batch, sequence_len, d_model) -> (batch, sequence_len, h, d_k) -> (batch, h, sequence_len, d_k)
        query = query.view(query.shape[0], query.shape[1], module._h, module._d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], module._h, module._d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], module._h, module._d_k).transpose(1, 2)

        x, attention_scores = MultiheadAttention.attention(query, key, value, mask, module._dropout)

        # Store the length of the words in the sequence that are active
        # mask_cpu = mask.detach().cpu()
        # indices = torch.where(mask_cpu[0][0][0] == 1)
        # active_mask = len(indices[0])

        attn_head_in = dict(
            # q, k, v have the following shape:
            # (1, seq_len, d_model) => (1, 350, 512)
            q = q.detach().cpu().numpy(), 
            k = k.detach().cpu().numpy(),
            v = v.detach().cpu().numpy(),

            # scalar
            # active_mask = active_mask,
            # This has shape (1, 1, 1, seq_len) => (1, 1, 1, 350)
            mask = mask.detach().cpu(),

            # query, key, value have the following shape:
            # (1, h, seq_len, d_k) => (1, 8, 350, 64)
            query = query.detach().cpu().numpy(),
            key = key.detach().cpu().numpy(),
            value = value.detach().cpu().numpy(),

            # Scores have the shape (1, h, seq_len, seq_len)
            # For this model it is (1, 8, 350, 350)
            attention_scores = attention_scores.detach().cpu().numpy()
        )

        # Shape of the output is (1, seq_len, d_model) => (1, 350, 512)
        attn_head_out = output.detach().cpu().numpy()

        return attn_head_in, attn_head_out

    def __process_feedforward_hook(self, module, input, output) -> Tuple[np.ndarray, np.ndarray]:
        in_val = input[0].detach().cpu().numpy()        # shape = (1, seq_len, d_model)
        out_val = output.detach().cpu().numpy()         # shape = (1, seq_len, d_model)

        return in_val, out_val

    def enc_embedding_hook(self, module, input, output) -> None:
        # input[0].shape = (1 ,seq_len, d_model) => (1, 350, 512)
        # output.shape = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder's embedding hook called")

        # Common processing for all embedding layer hooks
        in_val, out_val = self.__process_embedding_hook(module, input, output)

        # Store the embedding layer probe in memory
        self._enc_embedding_probe.add_probe(self._input_count, in_val, out_val)


    def enc0_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[3].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # output.shape = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder layer 0 attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._enc_0_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)


    def enc0_feedforward_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # output.shape   = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder layer 0 feedforward hook called")

        # Common processing for all feedforward layer hooks
        in_val, out_val = self.__process_feedforward_hook(module, input, output)

        # Store the feedforward layer probe in memory
        self._enc_0_feedforward_probe.add_probe(self._input_count, in_val, out_val)


    def enc5_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[3].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # output.shape = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder layer 5 attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._enc_5_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)


    def enc5_feedforward_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # output.shape   = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder layer 5 feedforward hook called")

        # Common processing for all feedforward layer hooks
        in_val, out_val = self.__process_feedforward_hook(module, input, output)

        # Store the feedforward layer probe in memory
        self._enc_5_feedforward_probe.add_probe(self._input_count, in_val, out_val)


    def encoder_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # output.shape   = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder hook called")

        in_val = input[0].detach().cpu().numpy()
        out_val = output.detach().cpu().numpy()

        # Store the encoder block's input and output probes in memory
        self._encoder_probe.add_probe(self._input_count, in_val, out_val)


    def dec_embedding_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x)  where x=1..seq_len
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder embedding hook called")

        # Common processing for all embedding layer hooks
        in_val, out_val = self.__process_embedding_hook(module, input, output)

        # Store the embedding layer probe in memory
        self._dec_embedding_probe.add_probe(self._input_count, in_val, out_val)


    def dec0_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, x, d_model)
        # input[2].shape = (1, x, d_model)
        # input[3].shape = (1, x, x)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 0 attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._dec_0_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)

        
    def dec0_cross_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, seq_len, d_model)
        # input[3].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 0 cross-attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._dec_0_cross_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)
        
    def dec0_feedforward_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 0 feedforward hook called")

        # Common processing for all feedforward layer hooks
        in_val, out_val = self.__process_feedforward_hook(module, input, output)

        # Store the feedforward layer probe in memory
        self._dec_0_feedforward_probe.add_probe(self._input_count, in_val, out_val)
        
    def dec5_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, x, d_model)
        # input[2].shape = (1, x, d_model)
        # input[3].shape = (1, x, x)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 5 attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._dec_5_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)
        
    def dec5_cross_attention_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, seq_len, d_model)
        # input[3].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 5 cross-attention hook called")

        # Common processing for all multi-attention hooks
        attn_head_in, attn_head_out = self.__process_attention_hook(module, input, output)

        # Store the multi-attention head probe in memory
        self._dec_5_cross_attn_probe.add_probe(self._input_count, attn_head_in, attn_head_out)

        
    def dec5_feedforward_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder layer 5 feedforward hook called")

        # Common processing for all feedforward layer hooks
        in_val, out_val = self.__process_feedforward_hook(module, input, output)

        # Store the feedforward layer probe in memory
        self._dec_5_feedforward_probe.add_probe(self._input_count, in_val, out_val)
        

    def decoder_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # input[3].shape = (1, x, x)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder hook called")
        
        in_val = input[0].detach().cpu().numpy()
        out_val = output.detach().cpu().numpy()

        # Store the encoder block's input and output probes in memory
        self._decoder_probe.add_probe(self._input_count, in_val, out_val)

    def projection_hook(self, module, input, output) -> None:
        # input[0].shape = (1, d_model) => (1, 512)
        # output.shape = (1, vocab_size) => (1, 30000)
        # print("Projection layer hook called")

        in_val = input[0].detach().cpu().numpy()
        out_val = output.detach().cpu().numpy()
        max_index = np.argmax(out_val)
        self._projection_probe.collect_probe(self._input_count, in_val, max_index)
