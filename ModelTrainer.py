
# Implementation of the Transmormer model trainer
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import torchmetrics.text
from torch.utils.tensorboard import SummaryWriter

# Definitions of the classes implemented by this application
from Transformer import Transformer


class ModelTrainer:
    def __init__(self, device : torch.device, model : Transformer) -> None:
        super().__init__()
        self._device = device
        self._model = model
        self._train_dataloader = None
        self._val_dataloader = None
        self._src_tokenizer = None
        self._tgt_tokenizer = None
        self._optimizer = None
        self._global_step = 0
        self._starting_epoch = 0
        self._seq_len = 0
        self._batch_iterator = None
        self._writer = None

    # Generate a causal mask
    def __causal_mask(self, size : int) -> torch.Tensor:
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    # Method to decode using a single encoder output
    def __greedy_decode(self, source, source_mask):
        sos_idx = self._tgt_tokenizer.token_to_id('[SOS]')
        eos_idx = self._tgt_tokenizer.token_to_id('[EOS]')

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
    def __validate(self) -> None:
        source_texts = list()
        expected = list()
        predicted = list()

        self._model.eval()
        count = 0

        with torch.no_grad():
            for batch in self._val_dataloader:
                encoder_input = batch['encoder_input'].to(self._device) # (batch_len, seq_len)
                encoder_mask = batch['encoder_mask'].to(self._device) # (batch_len, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = self.__greedy_decode(encoder_input, encoder_mask)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = self._tgt_tokenizer.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                
                # Print the source, target and model output
                self._batch_iterator.write('-'*80)
                self._batch_iterator.write(f"{f'SOURCE: ':>12}{source_text}")
                self._batch_iterator.write(f"{f'TARGET: ':>12}{target_text}")
                self._batch_iterator.write(f"{f'PREDICTED: ':>12}{model_out_text}\n")

                if count == 2:
                    self._batch_iterator.write('-'*80)
                    break

                count += 1

        if self._writer:
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.text.CharErrorRate()
            cer = metric(predicted, expected)
            self._writer.add_scalar('validation cer', cer, self._global_step)
            self._writer.flush()

            # Compute the word error rate
            metric = torchmetrics.text.WordErrorRate()
            wer = metric(predicted, expected)
            self._writer.add_scalar('validation wer', wer, self._global_step)
            self._writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.text.BLEUScore()
            bleu = metric(predicted, expected)
            self._writer.add_scalar('validation BLEU', bleu, self._global_step)
            self._writer.flush()


    # Preload the model weights from a file on the disk
    def __preload_model_weights(self, model_dir : Path, preload_file : str) -> bool:
        if preload_file == 'latest':
            # Find the model file saved after the last epoch of training
            model_files = [f for f in os.listdir(model_dir) if (os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".pt"))]
            if len(model_files) == 0:
                return False

            # Sort the files in ascending order. The last file in the
            # sorted list is the latest model weights file that was saved
            model_files.sort()
            latest_model_file = model_files[-1]
            preload_file_path = model_dir / latest_model_file
        else:
            # Caller specified model weight filename
            preload_file_path = model_dir / preload_file
            if os.path.isfile(preload_file_path) is False:
                return False

        print(f"Preloading model file: {str(preload_file_path)}")
        state = torch.load(str(preload_file_path))
        self._model.load_state_dict(state['model_state_dict'])
        self._starting_epoch = state['epoch'] + 1
        self._optimizer.load_state_dict(state['optimizer_state_dict'])

        # Total number of training steps across all epochs
        self._global_step = state['global_step']

        return True

    # Model training method 
    def train(self, config : dict, ds_dict : dict) -> None:
        # Save the dataloader and tokenizer of the source and target datasets 
        # so that they can be used by other methods of the ModelTrainer class
        self._train_dataloader = ds_dict['train_dataloader']
        self._val_dataloader = ds_dict['val_dataloader']
        self._src_tokenizer = ds_dict['src_tokenizer']
        self._tgt_tokenizer = ds_dict['tgt_tokenizer']
        val_dataset = ds_dict["val_dataset"]

        # Get the model configuration parameters
        lr = config['lr']
        test_name = config['test_name']
        preload_file = config['preload']
        num_epochs = config['num_epochs']
        self._seq_len = config['seq_len']
        model_folder = config["model_dir"]
        dataset_folder = config["dataset_dir"]

        # Model optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, eps=1e-9)

        # Directory for loading/storing the model weights
        model_dir = Path(model_folder)

        # Directory for storing the model validation dataset
        dataset_dir = Path(dataset_folder)

        if model_dir.exists():
            if preload_file == None:
                # If the previous weights are not to be loaded
                # then remove the model directory since it contains
                # the weights from a previous training session
                shutil.rmtree(model_dir)
            else:
                if self.__preload_model_weights(model_dir, preload_file) is not True:
                    # Load of model weights failed. Print a message and delete the model directory if it exists
                    print("Unable to find or load model weights. Training the model from scratch ...")
                    shutil.rmtree(model_dir)

        # If the model or dataset directories do not exist, then create it.
        if model_dir.exists() == False:
            model_dir.mkdir(parents=True, exist_ok=True)

        # Only the validation dataset is saved here so that it can be used
        # for probing the Transformer after the training is complete
        if dataset_dir.exists() == False:
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard writer
        self._writer = SummaryWriter(test_name)

        # Loss function used for the training
        loss_fn = nn.CrossEntropyLoss(ignore_index=self._src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(self._device)

        for epoch in range(self._starting_epoch, num_epochs):
            torch.cuda.empty_cache()
            self._model.train()

            self._batch_iterator = tqdm(self._train_dataloader, desc=f"Processing Epoch {epoch:02d}")
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
                loss = loss_fn(proj_output.view(-1, self._tgt_tokenizer.get_vocab_size()), label.view(-1))
                self._batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                self._writer.add_scalar('train loss', loss.item(), self._global_step)
                self._writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

                self._global_step += 1

            # Run the validation step at the end of each epoch
            self.__validate()

            # Save the model at the end of each epoch
            model_filename = model_dir / f"transformer_epoch_{epoch:02d}.pt"
            model_save_dict = {
                'epoch'                : epoch,
                'model_state_dict'     : self._model.state_dict(),
                'optimizer_state_dict' : self._optimizer.state_dict(),
                'global_step'          : self._global_step
            }
            torch.save(model_save_dict, model_filename)

            # If this is the first training epoch then save the validation dataset
            dataset_filename = dataset_dir / "validation_data.pt"
            if epoch == 0:
                torch.save(val_dataset, dataset_filename)

        self._writer.close()
