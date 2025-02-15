# Implementation of the TransformerProbe class

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from typing import Tuple
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics.text
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

# Definitions of the classes implemented by this application
from Transformer import Transformer
from MultiheadAttention import MultiheadAttention
from BilingualDataset import BilingualDataset
from ProbeManager import ProbeManager

special_dataset = [
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

    },
    {
        "id" : 3,
        "translation" : 
        {
            "en" : "The dog and the cat ate all the food",
            "fr" : "Le chien et le chat ont mangé toute la nourriture"
        }
    },
    {
        "id" : 4,
        "translation" : 
        {
            "en" : "There is a dog in the car and a cat in the van",
            "fr" : "Il y a un chien dans la voiture et un chat dans le van"
        }
    },
    {
        "id" : 5,
        "translation" : 
        {
            "en" : "She can play the violin and the guitar",
            "fr" : "Elle sait jouer du violon et de la guitare"
        }
    },
    {
        "id" : 6,
        "translation" : 
        {
            "en" : "The table with four chairs is empty",
            "fr" : "La table avec quatre chaises est vide"
        }
    },
    {
        "id" : 7,
        "translation" : 
        {
            "en" : "The king is alive and the people are happy",
            "fr" : "Le roi est vivant et le peuple est heureux"
        }
    },
    {
        "id" : 8,
        "translation" : 
        {
            "en" : "The murderer was a woman, not a man",
            "fr" : "Le meurtrier était une femme, pas un homme"
        }
    }
]
    

class TransformerProbe:
    # Constants
    MAX_PROBE_INPUT_DATA = 50
    
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
        self._writer = None
        self._val_dataloader = None
        self._global_step = 0
        self._epoch_probe = True

        # Performance metrics
        self._cer_list = list()
        self._wer_list = list()
        self._bleu_list = list()

    def __create_probes(self, N_inputs) -> None:
        # Encoder probe objects
        self._enc_embedding_probe = ProbeManager(N_inputs)          # Encoder's embedding layer probe
        self._enc_0_attn_probe = ProbeManager(N_inputs)             # Encoder 0 attention layer probe
        self._enc_0_feedforward_probe = ProbeManager(N_inputs)      # Encoder 0 feedforward layer probe
        self._enc_5_attn_probe = ProbeManager(N_inputs)             # Encoder 5 attention layer probe
        self._enc_5_feedforward_probe = ProbeManager(N_inputs)      # Encoder 5 feedforward layer probe
        self._encoder_probe = ProbeManager(N_inputs)                # Encoder block's input and output probe

        # Decoder probe objects
        self._dec_embedding_probe = ProbeManager(N_inputs)          # Decoder's embedding layer probe
        self._dec_0_attn_probe = ProbeManager(N_inputs)             # Decoder 0 attention layer probe
        self._dec_0_cross_attn_probe = ProbeManager(N_inputs)       # Decoder 0 cross-attention layer probe
        self._dec_0_feedforward_probe = ProbeManager(N_inputs)      # Decoder 0 feedforward layer probe
        self._dec_5_attn_probe = ProbeManager(N_inputs)             # Decoder 5 attention layer probe
        self._dec_5_cross_attn_probe = ProbeManager(N_inputs)       # Decoder 5 cross-attention layer probe
        self._dec_5_feedforward_probe = ProbeManager(N_inputs)      # Decoder 5 feedforward layer probe
        self._decoder_probe = ProbeManager(N_inputs)                # Decoder block's input and output probe

        # Projection layer probe object 
        # (special probe to collect multiple inputs/outputs for each test input)
        self._projection_probe = ProbeManager(N_inputs, True)

    def __save_probes(self, epoch, probe_dir, probe_config) -> None:
        # Save the current epoch's encoder probes
        self._enc_embedding_probe.save(epoch, probe_dir, probe_config["enc_embed_layer"], self._epoch_probe)
        self._enc_embedding_probe.clear()
        self._enc_0_attn_probe.save(epoch, probe_dir, probe_config["enc_layer_0_attn"], self._epoch_probe)
        self._enc_0_attn_probe.clear()
        self._enc_0_feedforward_probe.save(epoch, probe_dir, probe_config["enc_layer_0_feedforward"], self._epoch_probe)
        self._enc_0_feedforward_probe.clear()
        self._enc_5_attn_probe.save(epoch, probe_dir, probe_config["enc_layer_5_attn"], self._epoch_probe)
        self._enc_5_attn_probe.clear()
        self._enc_5_feedforward_probe.save(epoch, probe_dir, probe_config["enc_layer_5_feedforward"], self._epoch_probe)
        self._enc_5_feedforward_probe.clear()
        self._encoder_probe.save(epoch, probe_dir, probe_config["enc_block"], self._epoch_probe) 
        self._encoder_probe.clear()

        # Save the current epoch's decoder probes
        self._dec_embedding_probe.save(epoch, probe_dir, probe_config["dec_embed_layer"], self._epoch_probe)
        self._dec_embedding_probe.clear()
        self._dec_0_attn_probe.save(epoch, probe_dir, probe_config["dec_layer_0_attn"], self._epoch_probe)
        self._dec_0_attn_probe.clear()
        self._dec_0_cross_attn_probe.save(epoch, probe_dir, probe_config["dec_layer_0_cross_attn"], self._epoch_probe)
        self._dec_0_cross_attn_probe.clear()
        self._dec_0_feedforward_probe.save(epoch, probe_dir, probe_config["dec_layer_0_feedforward"], self._epoch_probe)
        self._dec_0_feedforward_probe.clear()
        self._dec_5_attn_probe.save(epoch, probe_dir, probe_config["dec_layer_5_attn"], self._epoch_probe)
        self._dec_5_attn_probe.clear()
        self._dec_5_cross_attn_probe.save(epoch, probe_dir, probe_config["dec_layer_5_cross_attn"], self._epoch_probe)
        self._dec_5_cross_attn_probe.clear()
        self._dec_5_feedforward_probe.save(epoch, probe_dir, probe_config["dec_layer_5_feedforward"], self._epoch_probe)
        self._dec_5_feedforward_probe.clear()
        self._decoder_probe.save(epoch, probe_dir, probe_config["dec_block"], self._epoch_probe)
        self._decoder_probe.clear()

        # Save the current epoch's projection layer probe
        self._projection_probe.save(epoch, probe_dir, probe_config["proj_layer"], self._epoch_probe)
        self._projection_probe.clear()

    def __hook_modules(self) -> None:
        # Encoder hooks
        #
        self._enc_embed_hook_handle = self._model._source_embed.register_forward_hook(self.enc_embedding_hook)
        self._enc0_attn_hook_handle = self._model._encoder._layers[0]._self_attention.register_forward_hook(self.enc0_attention_hook)
        self._enc0_feedforward_hook_handle = self._model._encoder._layers[0]._feed_forward.register_forward_hook(self.enc0_feedforward_hook)
        self._enc5_attn_hook_handle = self._model._encoder._layers[5]._self_attention.register_forward_hook(self.enc5_attention_hook)
        self._enc5_feedforward_hook_handle = self._model._encoder._layers[5]._feed_forward.register_forward_hook(self.enc5_feedforward_hook)
        self._encoder_hook_handle = self._model._encoder.register_forward_hook(self.encoder_hook)

        # Decoder hooks
        self._dec_embed_hook_handle = self._model._target_embed.register_forward_hook(self.dec_embedding_hook)
        self._dec0_attn_hook_handle = self._model._decoder._layers[0]._self_attention.register_forward_hook(self.dec0_attention_hook)
        self._dec0_cross_attn_hook_handle = self._model._decoder._layers[0]._cross_attention.register_forward_hook(self.dec0_cross_attention_hook)
        self._dec0_feedforward_hook_handle = self._model._decoder._layers[0]._feed_forward.register_forward_hook(self.dec0_feedforward_hook)
        self._dec5_attn_hook_handle = self._model._decoder._layers[5]._self_attention.register_forward_hook(self.dec5_attention_hook)
        self._dec5_cross_attn_hook_handle = self._model._decoder._layers[5]._cross_attention.register_forward_hook(self.dec5_cross_attention_hook)
        self._dec5_feedforward_hook_handle = self._model._decoder._layers[5]._feed_forward.register_forward_hook(self.dec5_feedforward_hook)
        self._decoder_hook_handle = self._model._decoder.register_forward_hook(self.decoder_hook)

        # Projection layer hook
        self._projection_hook_handle = self._model._projection.register_forward_hook(self.projection_hook)

        
    def __unhook_modules(self) -> None:
        # Encoder hook cleanup
        self._enc_embed_hook_handle.remove()
        self._enc0_attn_hook_handle.remove()
        self._enc0_feedforward_hook_handle.remove() 
        self._enc5_attn_hook_handle.remove()
        self._enc5_feedforward_hook_handle.remove() 
        self._encoder_hook_handle.remove() 

        # Decoder hook cleanup
        self._dec_embed_hook_handle.remove()
        self._dec0_attn_hook_handle.remove()
        self._dec0_cross_attn_hook_handle.remove()
        self._dec0_feedforward_hook_handle.remove()
        self._dec5_attn_hook_handle.remove()
        self._dec5_cross_attn_hook_handle.remove()
        self._dec5_feedforward_hook_handle.remove()
        self._decoder_hook_handle.remove()

        # Projection hook cleanup
        self._projection_hook_handle.remove()

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

    # Only train for one epoch and dump the probes after every batch
    def train_probe(self, model : Transformer, config : dict, probe_config: dict, ds_dict : dict) -> None:
        # Indicate that the probes are batch probes
        self._epoch_probe = False

        # Save the dataloader and tokenizer of the source and target datasets 
        # so that they can be used by other methods of the ModelTrainer class
        self._train_dataloader = ds_dict['train_dataloader']
        self._val_dataloader = ds_dict['val_dataloader']
        self._tokenizer_src = ds_dict['src_tokenizer']
        self._tokenizer_tgt = ds_dict['tgt_tokenizer']

        # Get the model configuration parameters
        lr = config['lr']
        self._seq_len = config['seq_len']
        self._d_model = config["d_model"]
        lang_src = config["lang_src"]
        lang_tgt = config["lang_tgt"]
        probe_folder = config["train_probe_dir"]
        use_special_dataset = config["use_special_dataset"] 
        train_probe_count = config["train_probe_count"] 

        # If the probe dir exists clean the directory by removing all
        # subdirectories
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
        
        # Save the model to a member variable so that it can be
        # accessed by other methods of this class
        self._model = model
        
        # Tensorboard writer
        log_dir = probe_dir / "runs"
        self._writer = SummaryWriter(log_dir)

        if use_special_dataset == 1:
            # Specially crafted dataset
            val_ds = BilingualDataset(special_dataset, self._tokenizer_src, self._tokenizer_src, lang_src, lang_tgt, self._seq_len)
            self._val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # Instantiate the probe classes which will in turn initialize the 
        # memory for storing the probes
        N_inputs = min(self.MAX_PROBE_INPUT_DATA, len(self._val_dataloader))
        # N_inputs = len(self._val_dataloader)
        self.__create_probes(N_inputs)

        # Model optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, eps=1e-9)

        # Loss function used for the training
        loss_fn = nn.CrossEntropyLoss(ignore_index=self._tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(self._device)

        # Calculate the cadence for the dumping of the probes and
        # initialize a counter for the probe cadence
        probe_cadence = len(self._train_dataloader) // train_probe_count
        batch_counter = 1
        batch_probe_counter = 0

        self._batch_iterator = tqdm(self._train_dataloader, desc=f"Processing Epoch 0:")
        self._model.train()

        # Train for one epoch over several batches
        for batch in self._batch_iterator:
            encoder_input = batch['encoder_input'].to(self._device) # (batch_len, seq_len)
            decoder_input = batch['decoder_input'].to(self._device) # (batch_len, seq_len)
            encoder_mask = batch['encoder_mask'].to(self._device) # (batch_len, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(self._device) # (batch_len, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = self._model.encode(encoder_input, encoder_mask) # (batch_len, seq_len, d_model)
            decoder_output = self._model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_len, seq_len, d_model)
            proj_output = self._model.project(decoder_output) # (batch_len, seq_len, vocab_size)
    
            # Compare the output with the label
            label = batch['label'].to(self._device) # (batch_len, seq_len)

            # Compute the cross-entropy loss
            loss = loss_fn(proj_output.view(-1, self._tokenizer_tgt.get_vocab_size()), label.view(-1))
            self._batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            self._writer.add_scalar('train loss', loss.item(), batch_probe_counter)
            self._writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            if batch_counter == probe_cadence:
                # Hook the layers of the model
                self.__hook_modules()

                # Run the validation step to collect the probes
                self.__validate(0)

                # Save the probes collected during the validation step
                self.__save_probes(batch_probe_counter, probe_dir, probe_config)

                # Unhook all the modules to continue training with the next batch
                self.__unhook_modules()

                # Reset the batch counter
                batch_counter = 1

                # Empty the GPU cache after validation and go back to training
                # the model
                torch.cuda.empty_cache()
                self._model.train()
            else:
                batch_counter += 1

        self._writer.close()


    def run(self, model : Transformer, config : dict, probe_config : dict) -> None:
        # Indicate that the probes are epoch probes
        self._epoch_probe = True

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

        key = "dataset_folder"
        if key in config:
            dataset_folder = config[key]
        else:
            dataset_folder = None

        lang_src = config["lang_src"]
        lang_tgt = config["lang_tgt"]
        self._seq_len = config["seq_len"]
        self._d_model = config["d_model"]
        N_epochs = config["num_epochs"]
        datasource = config["datasource"]
        use_special_dataset = config["use_special_dataset"]

        # Directory for loading the model weights
        model_dir = Path(model_folder)
        if model_dir.exists() == False:
            RuntimeError("TransformerProbe.run(): Model directory not found!")
            return

        # Save the model to a member variable so that it can be
        # accessed by other methods of this class
        self._model = model

        # Hook the specific layers of the model
        self.__hook_modules()

        # Create the Bilingual dataset from the raw dataset (batch size is 1 for the validation run)
        if use_special_dataset == 1:
            # Specially crafted dataset
            val_ds = BilingualDataset(special_dataset, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, self._seq_len)
        else:
            if dataset_folder:
                dataset_dir = Path(dataset_folder)
                if dataset_dir.exists() == False:
                    raise ValueError(f"Dataset directory {str(dataset_dir)} does not exist")

                dataset_fname = dataset_dir / "validataion_dataset.pt"
                if dataset_fname.exists():
                    valid_ds_raw = torch.load(dataset_fname)
                else:
                    raise ValueError(f"Dataset file {str(dataset_fname)} does not exist")
            else:
                # Use the Huggingface dataset
                ds_raw = load_dataset(datasource, f"{lang_src}-{lang_tgt}", split='train')

                # Split the dataset as training and validation. We will only use the validation dataset
                ds_raw_len = len(ds_raw)
                train_ds_size = ds_raw_len - self.MAX_PROBE_INPUT_DATA
                val_ds_size = self.MAX_PROBE_INPUT_DATA
                train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

            val_ds = BilingualDataset(val_ds_raw, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, self._seq_len)

        self._val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # If it exists clean the directory by removing all
        # subdirectories
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

        # Instantiate the probe classes which will in turn initialize the 
        # memory for storing the probes
        N_inputs = min(self.MAX_PROBE_INPUT_DATA, len(self._val_dataloader))
        # N_inputs = len(self._val_dataloader)
        self.__create_probes(N_inputs)

        # Tensorboard writer
        log_dir = probe_dir / "runs"
        self._writer = SummaryWriter(log_dir)

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

            # Save the probes for the current epoch
            self.__save_probes(epoch, probe_dir, probe_config)
            
        # Hook cleanup
        self.__unhook_modules()

        # Close the Tensorboard summary file
        if self._writer:
            self._writer.close()

        # Plot the performance metrics
        show_plot = False
        if show_plot:
            x = list(range(1, len(self._cer_list)+1))
            fig, ax = plt.subplots(1, 3)
            ax[0].plot(x, self._cer_list)
            ax[0].set_title('Character errors vs epoch')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Error')
            ax[0].grid(True)

            ax[1].plot(x, self._wer_list)
            ax[1].set_title('Word errors vs epoch')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Error')
            ax[1].grid(True)

            ax[2].plot(x, self._bleu_list)
            ax[2].set_title('BLEU score vs epoch')
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Score')
            ax[2].grid(True)
            plt.show()



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
                encoder_mask = batch['encoder_mask'].to(self._device)   # (batch_len, 1, 1, seq_len)

                # Strore the target tokens and mask as these will get saved in the encoder probe
                self._label = batch['label']                            # (batch_len, seq_len)
                self._decoder_mask = batch['decoder_mask']              # (batch_len, 1, 1, seq_len)

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

                if self._input_count == (max_validn_count-1):
                    batch_iterator.write('-'*80)

                self._input_count += 1

        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        self._cer_list.append(cer)

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        self._wer_list.append(wer)

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        self._bleu_list.append(bleu)

        # Write to the Tensorboard summary file
        if self._writer:
            self._writer.add_scalar('validation cer', cer, self._global_step)
            self._writer.flush()

            self._writer.add_scalar('validation wer', wer, self._global_step)
            self._writer.flush()

            self._writer.add_scalar('validation BLEU', bleu, self._global_step)
            self._writer.flush()

    #
    # ------------------------------------ Hooks into the Transformer layers --------------------------------------------
    #
    def __extract_input_output(self, input, output) -> Tuple[np.ndarray, np.ndarray]:
        in_val = input[0].detach().cpu().numpy()
        out_val = output.detach().cpu().numpy()

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

        attn_head_in = dict(
            # q, k, v have the following shape:
            # (1, seq_len, d_model) => (1, 350, 512)
            q = q.detach().cpu().numpy(), 
            k = k.detach().cpu().numpy(),
            v = v.detach().cpu().numpy(),

            # scalar
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


    def enc_embedding_hook(self, module, input, output) -> None:
        # input[0].shape = (1 ,seq_len) => (1, 350)
        # output.shape = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder's embedding hook called")

        enc_embedding_in = dict(
            src_tokens = input[0].detach().cpu().numpy(),
            tgt_tokens = self._label.numpy(),                         # shape = (1, seq_len) 
            tgt_mask = np.squeeze(self._decoder_mask.numpy())[-1],    # shape = (seq_len,)
        )

        enc_embedding_out = output.detach().cpu().numpy()

        # Store the embedding layer probe in memory
        self._enc_embedding_probe.add_probe(self._input_count, enc_embedding_in, enc_embedding_out)


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
        in_val, out_val = self.__extract_input_output(input, output)

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
        in_val, out_val = self.__extract_input_output(input, output)

        # Store the feedforward layer probe in memory
        self._enc_5_feedforward_probe.add_probe(self._input_count, in_val, out_val)


    def encoder_hook(self, module, input, output) -> None:
        # input[0].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[1].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # output.shape   = (1, seq_len, d_model) => (1, 350, 512)
        # print("Encoder hook called")

        encoder_in = dict(
            x = input[0].detach().cpu().numpy(),
            mask = input[1].detach().cpu().numpy()
        )

        encoder_out = output.detach().cpu().numpy()

        # Store the encoder block's input and output probes in memory
        self._encoder_probe.add_probe(self._input_count, encoder_in, encoder_out)


    def dec_embedding_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x)  where x=1..seq_len
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder embedding hook called")

        # Common processing for all embedding layer hooks
        in_val, out_val = self.__extract_input_output(input, output)

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
        in_val, out_val = self.__extract_input_output(input, output)

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
        in_val, out_val = self.__extract_input_output(input, output)

        # Store the feedforward layer probe in memory
        self._dec_5_feedforward_probe.add_probe(self._input_count, in_val, out_val)
        

    def decoder_hook(self, module, input, output) -> None:
        # input[0].shape = (1, x, d_model) => (1, x, 512)
        # input[1].shape = (1, seq_len, d_model) => (1, 350, 512)
        # input[2].shape = (1, 1, 1, seq_len) => (1, 1, 1, 350)
        # input[3].shape = (1, x, x)
        # output.shape = (1, x, d_model) => (1, x, 512)
        # print("Decoder hook called")

        decoder_in = dict(
            decoder_in = input[0].detach().cpu().numpy(),
            encoder_out = input[1].detach().cpu().numpy(),
            src_mask = input[2].detach().cpu().numpy(),
            tgt_mask = input[3].detach().cpu().numpy()
        )

        decoder_out = output.detach().cpu().numpy()    

        # Store the decoder block's input and output probes in memory
        self._decoder_probe.add_probe(self._input_count, decoder_in, decoder_out)


    def projection_hook(self, module, input, output) -> None:
        # input[0].shape = (1, d_model) => (1, 512)
        # output.shape = (1, vocab_size) => (1, 30000)
        # print("Projection layer hook called")

        in_val = input[0].detach().cpu().numpy()
        out_val = output.detach().cpu().numpy()
        max_index = np.argmax(out_val)
        self._projection_probe.collect_probe(self._input_count, in_val, max_index)
