
import sys
sys.path.append('..')
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
from sklearn.decomposition import PCA

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

class MultiheadAttentionAnalyzer(TransformerAnalyzer) :
    def __init__(self, model_config: dict, probe_config: dict) -> None:
        super().__init__(model_config, probe_config)

        # Number of attention layers in the encoder and decoder
        self._N_attention_layers = 6

        # Set the probe analysis function callback
        # member variable inherited from the parent class
        self._analyze_probes = self.__process_probes

    def __get_sentence_tokens(self, sentence_id : int) -> Tuple[int, np.array, int, np.array]:
        # Get the number of tokens in this source sentence
        # from the encoder block's input mask probe
        src_mask=self._encoder_probe._probe_in[sentence_id]["mask"]
        N_src_tokens = np.count_nonzero(np.squeeze(src_mask))

        # Get the encoder's embedding layer input and output probes
        enc_embedding_input = self._enc_embedding_probe._probe_in[sentence_id]
        enc_embedding_output = self._enc_embedding_probe._probe_out[sentence_id]

        # From the encoder's embedding layer input probe get the source and target tokens
        src_tokens = enc_embedding_input["src_tokens"]
        src_sentence_tokens = np.squeeze(src_tokens)[:N_src_tokens]

        tgt_mask=enc_embedding_input["tgt_mask"]
        N_tgt_tokens = np.count_nonzero(np.squeeze(tgt_mask))
        tgt_tokens = enc_embedding_input["tgt_tokens"]
        tgt_sentence_tokens = np.squeeze(tgt_tokens)[:N_tgt_tokens]

        return N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens

    def __pca_dimensionality_reduction(self, X : np.array, Y : np.array) -> Tuple[np.array, np.array]:
        # Retain 95% of the variance
        pca_X = PCA(0.95)
        pca_X.fit(X)
        X_n_components = pca_X.n_components_

        pca_Y = PCA(0.95)
        pca_Y.fit(Y)
        Y_n_components = pca_Y.n_components_

        # Get the maximum number of components to retain
        N_components = max(X_n_components, Y_n_components)

        # Fit the PCA model to the data and transform the data.
        # Do the inverse transform to get the original data back with reduced dimensions
        pca_X = PCA(n_components=N_components)
        # X_reduced = pca_X.fit_transform(X)
        transf_data = pca_X.fit_transform(X)
        X_reduced = pca_X.inverse_transform(transf_data)

        pca_Y = PCA(n_components=N_components)
        # Y_reduced = pca_Y.fit_transform(Y)
        transf_data = pca_Y.fit_transform(Y)
        Y_reduced = pca_Y.inverse_transform(transf_data)

        return X_reduced, Y_reduced

    def __compute_matrix_mi(self, X : np.array, Y : np.array, N_tokens : int) -> None:
        # X_reduced, Y_reduced = self.__pca_dimensionality_reduction(X, Y)
        i = np.arange(0, N_tokens, 1)
        j = np.arange(0, N_tokens, 1)
        i_pos, j_pos = np.meshgrid(i, j)
        ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

        # Zero out the mutual information matrix
        query_MI = np.zeros((N_tokens, N_tokens))

        for i, j in ij_pos:
            X_row = X[i]
            Y_row = Y[i]

            self._MI_estimator.set_inputs(X_row, Y_row)
            # MI_data = self._MI_estimator.kraskov_MI()
            # query_MI[x, y] = MI_data["MI"]
            MI, _ = self._MI_estimator.MINE_MI()
            query_MI[i, j] = MI

        MI_dict = dict(x_pos=i_pos, y_pos=j_pos, MI=query_MI)
        return MI_dict

    def __compute_head_mi(self, query_head, key_head, value_head, seq_len) -> dict:
        # Contains the mutual information of the input x and the 
        # query, key and value arrays for each attention layer
        query_key_MI_list = list()
        query_value_MI_list = list()
        key_value_MI_list = list()

        for x in range(seq_len):
            X = query_head[x]
            Y = key_head[x]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            # MI = self._MI_estimator.MINE_MI()
            query_key_MI_list.append(MI)

            X = query_head[x]
            Y = value_head[x]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            #MI = self._MI_estimator.MINE_MI()
            query_value_MI_list.append(MI)

            X = key_head[x]
            Y = value_head[x]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            #MI = self._MI_estimator.MINE_MI()
            key_value_MI_list.append(MI)

        mi_dict = dict(QK_mi_list=query_key_MI_list, QV_mi_list=query_value_MI_list, KV_mi_list=key_value_MI_list)
        return mi_dict  


    def __get_query_key_value(self, sentence_id : int, N_src_tokens : int):
        # Attention layer 0
        enc_attn_input = self._enc_0_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_0 = x[:N_src_tokens]
        query_0 = query[:N_src_tokens]
        key_0 = key[:N_src_tokens]
        value_0 = value[:N_src_tokens]

        # Attention layer 1
        enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_1 = x[:N_src_tokens]
        query_1 = query[:N_src_tokens]
        key_1 = key[:N_src_tokens]
        value_1 = value[:N_src_tokens]
        
        # Attention layer 2
        enc_attn_input = self._enc_2_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_2 = x[:N_src_tokens]
        query_2 = query[:N_src_tokens]
        key_2 = key[:N_src_tokens]
        value_2 = value[:N_src_tokens]

        # Attention layer 3
        enc_attn_input = self._enc_3_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_3 = x[:N_src_tokens]
        query_3 = query[:N_src_tokens]
        key_3 = key[:N_src_tokens]
        value_3 = value[:N_src_tokens]

        # Attention layer 4
        enc_attn_input = self._enc_4_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_4 = x[:N_src_tokens]
        query_4 = query[:N_src_tokens]
        key_4 = key[:N_src_tokens]
        value_4 = value[:N_src_tokens]

        # Attention layer 5
        enc_attn_input = self._enc_5_attn_probe._probe_in[sentence_id]
        x = enc_attn_input["q"].squeeze()
        query = enc_attn_input["query"].squeeze()
        key = enc_attn_input["key"].squeeze()
        value = enc_attn_input["value"].squeeze()
        x_5 = x[:N_src_tokens]
        query_5 = query[:N_src_tokens]
        key_5 = key[:N_src_tokens]
        value_5 = value[:N_src_tokens]

        QKV_dict = { 'attention_0': {"x": x_0, "query": query_0, "key": key_0, "value": value_0},
                     'attention_1': {"x": x_1, "query": query_1, "key": key_1, "value": value_1},
                     'attention_2': {"x": x_2, "query": query_2, "key": key_2, "value": value_2},
                     'attention_3': {"x": x_3, "query": query_3, "key": key_3, "value": value_3},
                     'attention_4': {"x": x_4, "query": query_4, "key": key_4, "value": value_4},
                     'attention_5': {"x": x_5, "query": query_5, "key": key_5, "value": value_5} }

        return QKV_dict 

    def __get_head_query_key_value(self, sentence_id : int, N_src_tokens : int = None):
    # Attention layer 0
        enc_attn_input = self._enc_0_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()

        if N_src_tokens is None:
            N_src_tokens = query.shape[0]

        query_0 = query[:,:N_src_tokens]
        key_0 = key[:,:N_src_tokens]
        value_0 = value[:,:N_src_tokens]

        # Attention layer 1
        enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()
        query_1 = query[:,:N_src_tokens]
        key_1 = key[:,:N_src_tokens]
        value_1 = value[:,:N_src_tokens]
        
        # Attention layer 2
        enc_attn_input = self._enc_2_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()
        query_2 = query[:,:N_src_tokens]
        key_2 = key[:,:N_src_tokens]
        value_2 = value[:,:N_src_tokens]

        # Attention layer 3
        enc_attn_input = self._enc_3_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()
        query_3 = query[:,:N_src_tokens]
        key_3 = key[:,:N_src_tokens]
        value_3 = value[:,:N_src_tokens]

        # Attention layer 4
        enc_attn_input = self._enc_4_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()
        query_4 = query[:,:N_src_tokens]
        key_4 = key[:,:N_src_tokens]
        value_4 = value[:,:N_src_tokens]

        # Attention layer 5
        enc_attn_input = self._enc_5_attn_probe._probe_in[sentence_id]
        query = enc_attn_input["query_head"].squeeze()
        key = enc_attn_input["key_head"].squeeze()
        value = enc_attn_input["value_head"].squeeze()
        query_5 = query[:,:N_src_tokens]
        key_5 = key[:,:N_src_tokens]
        value_5 = value[:,:N_src_tokens]

        QKV_dict = { 'attention_0': {"query": query_0, "key": key_0, "value": value_0},
                     'attention_1': {"query": query_1, "key": key_1, "value": value_1},
                     'attention_2': {"query": query_2, "key": key_2, "value": value_2},
                     'attention_3': {"query": query_3, "key": key_3, "value": value_3},
                     'attention_4': {"query": query_4, "key": key_4, "value": value_4},
                     'attention_5': {"query": query_5, "key": key_5, "value": value_5} }

        return QKV_dict 

    def __get_attention_scores(self, sentence_id, N_src_tokens, attention_layer) -> np.array:
        match attention_layer:
            case 0:
                enc_attn_input = self._enc_0_attn_probe._probe_in[sentence_id]

            case 1:
                enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]

            case 2:
                enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]

            case 3:
                enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]

            case 4:
                enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]

            case 5:
                enc_attn_input = self._enc_1_attn_probe._probe_in[sentence_id]

            case _:
                print(f"Invalid attention layer {attention_layer}")
                return

        attention_scores = enc_attn_input["attention_scores"].squeeze()
        return attention_scores[:, :N_src_tokens, :N_src_tokens]


    def __stack_QKV_arrays(self, QKV_list, N_inputs):
        # This will contain the query, key and value arrays
        # of all the input sentences of this epoch
        x_0_array = None; query_0_array = None;  key_0_array = None;  value_0_array = None
        x_1_array = None; query_1_array = None;  key_1_array = None;  value_1_array = None
        x_2_array = None; query_2_array = None;  key_2_array = None;  value_2_array = None
        x_3_array = None; query_3_array = None;  key_3_array = None;  value_3_array = None
        x_4_array = None; query_4_array = None;  key_4_array = None;  value_4_array = None
        x_5_array = None; query_5_array = None;  key_5_array = None;  value_5_array = None

        for i in range(N_inputs):
            x_0 = QKV_list[i]['attention_0']["x"]
            query_0 = QKV_list[i]['attention_0']["query"]
            key_0 = QKV_list[i]['attention_0']["key"]
            value_0 = QKV_list[i]['attention_0']["value"]
            x_1 = QKV_list[i]['attention_1']["x"]
            query_1 = QKV_list[i]['attention_1']["query"]
            key_1 = QKV_list[i]['attention_1']["key"]
            value_1 = QKV_list[i]['attention_1']["value"]
            x_2 = QKV_list[i]['attention_2']["x"]
            query_2 = QKV_list[i]['attention_2']["query"]
            key_2 = QKV_list[i]['attention_2']["key"]
            value_2 = QKV_list[i]['attention_2']["value"]
            x_3 = QKV_list[i]['attention_3']["x"]
            query_3 = QKV_list[i]['attention_3']["query"]
            key_3 = QKV_list[i]['attention_3']["key"]
            value_3 = QKV_list[i]['attention_3']["value"]
            x_4 = QKV_list[i]['attention_4']["x"]
            query_4 = QKV_list[i]['attention_4']["query"]
            key_4 = QKV_list[i]['attention_4']["key"]
            value_4 = QKV_list[i]['attention_4']["value"]
            x_5 = QKV_list[i]['attention_5']["x"]
            query_5 = QKV_list[i]['attention_5']["query"]
            key_5 = QKV_list[i]['attention_5']["key"]
            value_5 = QKV_list[i]['attention_5']["value"]

            if query_0_array is None:
                x_0_array = x_0
                query_0_array = query_0
                key_0_array = key_0
                value_0_array = value_0
            else:
                x_0_array = np.vstack((x_0_array, x_0))
                query_0_array = np.vstack((query_0_array, query_0))
                key_0_array = np.vstack((key_0_array, key_0))
                value_0_array = np.vstack((value_0_array, value_0))

            if query_1_array is None:
                x_1_array = x_1
                query_1_array = query_1
                key_1_array = key_1
                value_1_array = value_1 
            else:
                x_1_array = np.vstack((x_1_array, x_1))
                query_1_array = np.vstack((query_1_array, query_1))
                key_1_array = np.vstack((key_1_array, key_1))
                value_1_array = np.vstack((value_1_array, value_1))

            if query_2_array is None:
                x_2_array = x_2
                query_2_array = query_2
                key_2_array = key_2
                value_2_array = value_2
            else:
                x_2_array = np.vstack((x_2_array, x_2))
                query_2_array = np.vstack((query_2_array, query_2))
                key_2_array = np.vstack((key_2_array, key_2))
                value_2_array = np.vstack((value_2_array, value_2))

            if query_3_array is None:
                x_3_array = x_3
                query_3_array = query_3
                key_3_array = key_3
                value_3_array = value_3
            else:
                x_3_array = np.vstack((x_3_array, x_3))
                query_3_array = np.vstack((query_3_array, query_3))
                key_3_array = np.vstack((key_3_array, key_3))
                value_3_array = np.vstack((value_3_array, value_3))

            if query_4_array is None:
                x_4_array = x_4
                query_4_array = query_4
                key_4_array = key_4
                value_4_array = value_4
            else:
                x_4_array = np.vstack((x_4_array, x_4))
                query_4_array = np.vstack((query_4_array, query_4))
                key_4_array = np.vstack((key_4_array, key_4))
                value_4_array = np.vstack((value_4_array, value_4))
                
            if query_5_array is None:
                x_5_array = x_5
                query_5_array = query_5
                key_5_array = key_5
                value_5_array = value_5
            else:
                x_5_array = np.vstack((x_5_array, x_5))
                query_5_array = np.vstack((query_5_array, query_5))
                key_5_array = np.vstack((key_5_array, key_5))
                value_5_array = np.vstack((value_5_array, value_5))

        QKV_dict = { 'attention_0': {"x": x_0_array, "query": query_0_array, "key": key_0_array, "value": value_0_array},
                     'attention_1': {"x": x_1_array, "query": query_1_array, "key": key_1_array, "value": value_1_array},
                     'attention_2': {"x": x_2_array, "query": query_2_array, "key": key_2_array, "value": value_2_array},
                     'attention_3': {"x": x_3_array, "query": query_3_array, "key": key_3_array, "value": value_3_array},
                     'attention_4': {"x": x_4_array, "query": query_4_array, "key": key_4_array, "value": value_4_array},
                     'attention_5': {"x": x_5_array, "query": query_5_array, "key": key_5_array, "value": value_5_array} }

        return QKV_dict 

    def __compute_QKV_matrix_mi(self, x, query, key, value, N_dimensions) -> dict:
        # These will contain the mutual information of the input x and the 
        # query, key and value arrays for each attention layer
        xQ_mi_list = list()
        xK_mi_list = list()
        xV_mi_list = list()

        for n in range(N_dimensions):
            X = x[:, n]
            Y = query[:, n]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            xQ_mi_list.append(MI)

            Y = key[:, n]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            xK_mi_list.append(MI)

            Y = value[:, n]
            self._MI_estimator.set_inputs(X, Y)
            MI = self._MI_estimator.kraskov_MI()["MI"]
            xV_mi_list.append(MI)

        mi_dict = dict(xQ_mi_list=xQ_mi_list, xK_mi_list=xK_mi_list, xV_mi_list=xV_mi_list)
        return mi_dict

    def __compute_entropy(self, query, key, value, N_dimensions) -> dict:
        # These will contain entropy for the query, key and value arrays
        # for each attention layer
        Q_entropy_list = list()
        K_entropy_list = list()
        V_entropy_list = list()

        for n in range(N_dimensions):
            Y = query[:, n]
            self._MI_estimator.set_inputs(Y, Y)
            H = self._MI_estimator.kraskov_entropy()
            Q_entropy_list.append(H)

            Y = key[:, n]
            self._MI_estimator.set_inputs(Y, Y)
            H = self._MI_estimator.kraskov_entropy()
            K_entropy_list.append(H)

            Y = value[:, n]
            self._MI_estimator.set_inputs(Y, Y)
            H = self._MI_estimator.kraskov_entropy()
            V_entropy_list.append(H)

        entropy_dict = dict(Q_entropy_list=Q_entropy_list, K_entropy_list=K_entropy_list, V_entropy_list=V_entropy_list)

        return entropy_dict

    def __compute_QKV_entropy_mi(self, QKV_dict) -> Tuple[dict, dict]:
        # This will contain the entropy values for the query, key and value arrays
        QKV_entropy_dict = dict()
        QKV_mi_dict = dict()

        # Get the dimensions of the model from the number
        # of columns of the query array
        x = QKV_dict['attention_0']["x"]
        N_dimensions = x.shape[1]

        for i in range(self._N_attention_layers):
            x = QKV_dict[f'attention_{i}']["x"]
            query = QKV_dict[f'attention_{i}']["query"]
            key = QKV_dict[f'attention_{i}']["key"]
            value = QKV_dict[f'attention_{i}']["value"]

            entropy_dict = self.__compute_entropy(query, key, value, N_dimensions)
            mi_dict = self.__compute_QKV_matrix_mi(x, query, key, value, N_dimensions)

            Q_entropy_list = entropy_dict["Q_entropy_list"]
            K_entropy_list = entropy_dict["K_entropy_list"]
            V_entropy_list = entropy_dict["V_entropy_list"]
            xQ_mi_list = mi_dict["xQ_mi_list"]
            xK_mi_list = mi_dict["xK_mi_list"]
            xV_mi_list = mi_dict["xV_mi_list"]

            QKV_entropy_dict[f'attention_{i}'] = {"query": Q_entropy_list, "key": K_entropy_list, "value": V_entropy_list}
            QKV_mi_dict[f'attention_{i}'] = {"query": xQ_mi_list, "key": xK_mi_list, "value": xV_mi_list}

        return QKV_entropy_dict, QKV_mi_dict

    def __compute_head_QKV_entropy(self, QKV_dict):
        # This will contain the entropy values for the query, key and value arrays
        QKV_entropy_dict = dict()

        # Get the dimensions of the model from the number
        # of columns of the query array
        query = QKV_dict['attention_0']["query"]
        N_heads = query.shape[0]
        N_dimensions = query.shape[2]

        for i in range(self._N_attention_layers):
            query = QKV_dict[f'attention_{i}']["query"]
            key = QKV_dict[f'attention_{i}']["key"]
            value = QKV_dict[f'attention_{i}']["value"]

            # Contains a list of lists. The number of elements 
            # in each of the following lists is equal to the 
            # number of attention heads. Each element in the list
            # is a list containing the entropy across dimensions 
            # of the corresponding head of the query, key or value array
            Q_head_entropy_list = list()
            K_head_entropy_list = list()
            V_head_entropy_list = list()

            for j in range(N_heads):
                query_head = query[j]
                key_head = key[j]
                value_head = value[j]

                entropy_dict = self.__compute_entropy(query_head, key_head, value_head, N_dimensions)

                # The dictionary contains the list of entropy values for the query, key and value 
                # attention head arrays across all dimensions
                Q_head_entropy_list.append(entropy_dict["Q_entropy_list"])
                K_head_entropy_list.append(entropy_dict["K_entropy_list"])
                V_head_entropy_list.append(entropy_dict["V_entropy_list"])

            QKV_entropy_dict[f'attention_{i}'] = {"query": Q_head_entropy_list, "key": K_head_entropy_list, "value": V_head_entropy_list}

        return QKV_entropy_dict

    def __compute_head_QKV_mi(self, QKV_dict):
        # This will contain the entropy values for the query, key and value arrays
        QKV_entropy_dict = dict()

        # Get the dimensions of the model from the number
        # of columns of the query array
        query = QKV_dict['attention_0']["query"]
        N_heads = query.shape[0]
        seq_len = query.shape[1]

        for i in range(self._N_attention_layers):
            query = QKV_dict[f'attention_{i}']["query"]
            key = QKV_dict[f'attention_{i}']["key"]
            value = QKV_dict[f'attention_{i}']["value"]

            # Contains a list of lists. The number of elements 
            # in each of the following lists is equal to the 
            # number of attention heads. Each element in the list
            # is a list containing the mutual information across the
            # seq_len dimension of the corresponding query and key heads
            QK_head_mi_list = list()
            QV_head_mi_list = list()
            KV_head_mi_list = list()

            for j in range(N_heads):
                query_head = query[j]
                key_head = key[j]
                value_head = value[j]

                entropy_dict = self.__compute_head_mi(query_head, key_head, value_head, seq_len)

                # The dictionary contains the list of mutual information values between the query and key 
                # attention head arrays across the seq_len dimension
                QK_head_mi_list.append(entropy_dict["QK_mi_list"])
                QV_head_mi_list.append(entropy_dict["QV_mi_list"])
                KV_head_mi_list.append(entropy_dict["KV_mi_list"])

            QKV_entropy_dict[f'attention_{i}'] = {"qk_mi": QK_head_mi_list, "qv_mi": QV_head_mi_list, "kv_mi": KV_head_mi_list}

        return QKV_entropy_dict


    def __plot_QKV_head_entropy(self, QKV_entropy_list, min_val, max_val, epochs_to_analyze, vector_str, plot_title=None):
        # Plot the entropy values for each dimension of the query/key/value array across epochs
        for atten_layer in range(self._N_attention_layers):
            attn_entropy_list = np.array(QKV_entropy_list[0][f'attention_{atten_layer}'][vector_str])
            N_heads = len(attn_entropy_list)
            fig, axs = plt.subplots(2, 4)

            for head in range(N_heads):
                head_entropy_array = None

                for epoch in range(len(epochs_to_analyze)):
                    epoch_head_entropy = np.array(QKV_entropy_list[epoch][f'attention_{atten_layer}'][vector_str][head])
                    if head_entropy_array is None:
                        head_entropy_array = epoch_head_entropy
                    else:
                        head_entropy_array = np.vstack((head_entropy_array, epoch_head_entropy))

                a = head//4
                b = head%4
                im = axs[a, b].imshow(head_entropy_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, head_entropy_array.shape[1]-1])
                axs[a, b].set_aspect('auto')
                axs[a, b].set_title(f"Head {head}")
                axs[a, b].set_xlabel("Epoch")
                if b == 0:
                    axs[a, b].set_ylabel("Dimension")
                axs[a, b].set_xticks([0, 4, 9, 14, 19])

            if plot_title:
                fig.suptitle(plot_title)
            else:
                fig.suptitle(f"Entropy of the {vector_str} head for attention layer {atten_layer}")

            plt.subplots_adjust(hspace=0.8, right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            # plt.tight_layout()

            plt.show(block=False)

    def __plot_QKV_head_mi(self, QKV_mi_list, min_val, max_val, N_src_tokens, epochs_to_analyze, selection, MI_str) -> None :
        # Plot the mutual information values for each dimension of the query/key/value array across epochs
        for atten_layer in range(self._N_attention_layers):
            attn_mi_list = np.array(QKV_mi_list[0][f'attention_{atten_layer}'][selection])
            N_heads = len(attn_mi_list)
            fig, axs = plt.subplots(2, 4)

            for head in range(N_heads):
                head_mi_array = None

                for epoch in range(len(epochs_to_analyze)):
                    epoch_head_mi = np.array(QKV_mi_list[epoch][f'attention_{atten_layer}'][selection][head])
                    if head_mi_array is None:
                        head_mi_array = epoch_head_mi
                    else:
                        head_mi_array = np.vstack((head_mi_array, epoch_head_mi))

                a = head//4
                b = head%4
                im = axs[a, b].imshow(head_mi_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, head_mi_array.shape[1]-1])
                axs[a, b].set_aspect('auto')
                axs[a, b].set_title(f"Head {head}")
                axs[a, b].set_xlabel("Epoch")
                if b == 0:
                    axs[a, b].set_ylabel("Token Position")
                axs[a, b].set_xticks([0, 4, 9, 14, 19])
                axs[a, b].set_yticks(np.arange(0, N_src_tokens, 2))

            fig.suptitle(f"Mutual Information of the {MI_str} head for attention layer {atten_layer}")
            plt.subplots_adjust(hspace=0.8, right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            plt.show(block=False)

    def __plot_QKV_matrix_entropy_mi(self, QKV_entropy_mi_list, min_val, max_val, epochs_to_analyze, vector_str, plot_title=None):
        # Plot the entropy values for each dimension of the query/key/value array across epochs
        fig, axs = plt.subplots(2, 3)
        for atten_layer in range(self._N_attention_layers):
            entropy_array = None

            for epoch in range(len(epochs_to_analyze)):
                epoch_entropy = np.array(QKV_entropy_mi_list[epoch][f'attention_{atten_layer}'][vector_str])
                if entropy_array is None:
                    entropy_array = epoch_entropy
                else:
                    entropy_array = np.vstack((entropy_array, epoch_entropy))

            a = atten_layer//3
            b = atten_layer%3
            im = axs[a, b].imshow(entropy_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, entropy_array.shape[1]-1])
            axs[a, b].set_aspect('auto')
            axs[a, b].set_title(f"Attention Layer {atten_layer}")
            axs[a, b].set_xlabel("Epoch")
            axs[a, b].set_ylabel("Dimension")
            axs[a, b].set_xticks(range(0, len(epochs_to_analyze)), epochs_to_analyze)


        if plot_title:
            fig.suptitle(plot_title)
        else:
            fig.suptitle(f"Entropy of each dimension of the {vector_str} array across epochs")

        plt.subplots_adjust(hspace=0.8, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show(block=False)

    def __plot_attention_scores(self, attention_scores, input_words, plot_title) -> None:
        N_heads = attention_scores.shape[0]

        fig, axs = plt.subplots(2, 4)
        for head in range(N_heads):
            a = head//4
            b = head%4
            attention_scores_matrix = attention_scores[head]
            # im = axs[a, b].imshow(attention_scores_matrix, cmap=plt.cm.Wistia, interpolation='none', origin='lower', extent=[0, attention_scores.shape[1]-1, 0, attention_scores.shape[2]-1])
            im = axs[a, b].imshow(attention_scores_matrix, cmap=plt.cm.Wistia, interpolation='none', origin='lower')

            # Add text annotations
            for i in range(attention_scores_matrix.shape[0]):
                for j in range(attention_scores_matrix.shape[1]):
                    text = axs[a, b].text(j, i, f"{attention_scores_matrix[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)

            axs[a, b].set_xticks(range(0, len(input_words)), input_words, rotation=45)
            axs[a, b].set_yticks(range(0, len(input_words)), input_words, rotation=45)
            axs[a, b].set_aspect('auto')
            axs[a, b].set_title(f"Head {head}")

        fig.suptitle(plot_title)

        plt.subplots_adjust(hspace=0.8, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show(block=False)


    def __get_min_max_QKV_matrix(self, QKV_list, epochs_to_analyze):
        # Initialization to extreme values
        min_val = 1e9
        max_val = -1e9

        for epoch in range(len(epochs_to_analyze)):
            for atten_layer in range(self._N_attention_layers):
                list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['query'])
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

                list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['key'])
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

                list_vals = np.array(QKV_list[epoch][f'attention_{atten_layer}']['value'])
                min_val = min(min_val, np.min(list_vals))
                max_val = max(max_val, np.max(list_vals))

        return min_val, max_val

    def __get_min_max_QKV_head(self, QKV_list, epochs_to_analyze, entropy = True):
        # Initialization to extreme values
        min_val = 1e9
        max_val = -1e9

        for epoch in range(len(epochs_to_analyze)):
            for atten_layer in range(self._N_attention_layers):
                sel_str = 'query' if entropy else 'qk_mi'
                list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
                for list_vals in list_of_lists:
                    min_val = min(min_val, np.min(list_vals))
                    max_val = max(max_val, np.max(list_vals))

                sel_str = 'key' if entropy else 'qv_mi'
                list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
                for list_vals in list_of_lists:
                    min_val = min(min_val, np.min(list_vals))
                    max_val = max(max_val, np.max(list_vals))

                sel_str = 'value' if entropy else 'kv_mi'
                list_of_lists = np.array(QKV_list[epoch][f'attention_{atten_layer}'][sel_str])
                for list_vals in list_of_lists:
                    min_val = min(min_val, np.min(list_vals))
                    max_val = max(max_val, np.max(list_vals))

        return min_val, max_val


    def __process_QKV_matrix(self):
        print("Computing the entropy and mutual information of each dimension of the Query, Key and Value arrays ...")

        epochs_to_analyze = np.arange(0, 20, 1)
        # epochs_to_analyze = [0, 4, 9, 14, 19]

        # These will contain entropy and mutual information values for the query, key and value arrays
        # for each attention layer for all epochs
        QKV_entropy_list = list()
        QKV_mi_list = list()

        # Analyze the probes of the Multihead Attention layers of the encoder for each epoch
        for epoch in epochs_to_analyze:
            # For this epoch, load all the encoder layer probe files from disk
            super().load_encoder_probes(epoch)

            # Number of input sentences in this epoch
            N_inputs = len(self._encoder_probe._probe_in)

            # This will contain the QKV dictionaries for all the attention layers
            # of all the input sentences of this epoch
            QKV_list = list()

            # Iterate across all the input sentences of this epoch and get the query, key and value arrays.
            # Stack the arrays horizontally after each iteration
            for i in range(N_inputs):
                N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = self. __get_sentence_tokens(i)
                # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
                # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

                # Get the query, key, value arrays for all the attention layers of this input sentence
                QKV_dict = self.__get_query_key_value(i, N_src_tokens)
                QKV_list.append(QKV_dict)


            # Concatenate the query, key and value arrays horizontally for all the input sentences of this epoch
            QKV_stacked_dict = self.__stack_QKV_arrays(QKV_list, N_inputs)

            # Compute the entropy and mutual information for each dimension of the query, key and value arrays
            QKV_entropy_dict, QKV_mi_dict = self.__compute_QKV_entropy_mi(QKV_stacked_dict)
            QKV_entropy_list.append(QKV_entropy_dict)
            QKV_mi_list.append(QKV_mi_dict)

        # Plot the entropy values for each dimension of the query array across epochs
        min_val, max_val = self.__get_min_max_QKV_matrix(QKV_entropy_list, epochs_to_analyze)
        self.__plot_QKV_matrix_entropy_mi(QKV_entropy_list, min_val, max_val, epochs_to_analyze, 'query')
        self.__plot_QKV_matrix_entropy_mi(QKV_entropy_list, min_val, max_val, epochs_to_analyze, 'key')
        self.__plot_QKV_matrix_entropy_mi(QKV_entropy_list, min_val, max_val, epochs_to_analyze, 'value')

        # Get the min and max mutual information values across all attention layers and all epochs
        min_val, max_val = self.__get_min_max_QKV_matrix(QKV_mi_list, epochs_to_analyze)

        # Plot the mutual information values for each dimension of the input and query array across epochs
        plot_title = "Mutual Information of each dimension of the encoder input and Query array across epochs"
        self.__plot_QKV_matrix_entropy_mi(QKV_mi_list, min_val, max_val, epochs_to_analyze, 'query', plot_title)

        # Plot the mutual information values for each dimension of the input and key array across epochs
        plot_title = "Mutual Information of each dimension of the encoder input and Key array across epochs"
        self.__plot_QKV_matrix_entropy_mi(QKV_mi_list, min_val, max_val, epochs_to_analyze, 'key', plot_title)

        # Plot the mutual information values for each dimension of the input and value array across epochs
        plot_title = "Mutual Information of each dimension of the encoder input and Value array across epochs"
        self.__plot_QKV_matrix_entropy_mi(QKV_mi_list, min_val, max_val, epochs_to_analyze, 'value', plot_title)


    def __process_QKV_heads(self):
        print("Computing the entropy and mutual information of each dimension of the Query, Key and Value heads ...")
        epochs_to_analyze = np.arange(0, 20, 1)
        # epochs_to_analyze = [0, 4, 9, 14, 19]

        # These will contain entropy and mutual information values for the query, key and value head arrays
        # for each attention layer for all epochs. Each element in these lists are a list of values for each head
        QKV_entropy_list = list()
        QKV_mi_list = list()

        # Input sentence ID
        sentence_id = 7

        # Analyze the probes of the Multihead Attention layers of the encoder for each epoch
        for epoch in epochs_to_analyze:
            # For this epoch, load all the encoder layer probe files from disk
            super().load_encoder_probes(epoch)

            # Get the number of tokens of the sentence
            N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = self. __get_sentence_tokens(sentence_id)
            # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
            # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

            # Get the query, key, value arrays for all the attention layers of this input sentence
            # N_src_tokens=350
            QKV_dict = self.__get_head_query_key_value(sentence_id, N_src_tokens)

            # Compute the entropy and mutual information for each dimension of the query, key and value arrays
            QKV_entropy_dict = self.__compute_head_QKV_entropy(QKV_dict)
            QKV_mi_dict = self.__compute_head_QKV_mi(QKV_dict)

            QKV_entropy_list.append(QKV_entropy_dict)
            QKV_mi_list.append(QKV_mi_dict)

        # Plot the entropy values for each dimension of the query array across epochs
        min_val, max_val = self.__get_min_max_QKV_head(QKV_entropy_list, epochs_to_analyze)
        self.__plot_QKV_head_entropy(QKV_entropy_list, min_val, max_val, epochs_to_analyze, "query")
        self.__plot_QKV_head_entropy(QKV_entropy_list, min_val, max_val, epochs_to_analyze, "key")
        self.__plot_QKV_head_entropy(QKV_entropy_list, min_val, max_val, epochs_to_analyze, "value")

        min_val, max_val = self.__get_min_max_QKV_head(QKV_mi_list, epochs_to_analyze, False)
        self.__plot_QKV_head_mi(QKV_mi_list, min_val, max_val, N_src_tokens, epochs_to_analyze, "qk_mi", "Query-Key")
        self.__plot_QKV_head_mi(QKV_mi_list, min_val, max_val, N_src_tokens, epochs_to_analyze, "qv_mi", "Query-Value")
        self.__plot_QKV_head_mi(QKV_mi_list, min_val, max_val, N_src_tokens, epochs_to_analyze, "kv_mi", "Key-Value")

    def __process_attention_scores(self):
        print("Computing the attention scores ...")
        # epochs_to_analyze = np.arange(0, 20, 1)
        # epochs_to_analyze = [0, 4, 9, 14, 19]

        # Input sentence ID
        sentence_id = 0
        epoch = 19
        attention_layer = 5

        # For this epoch, load all the encoder layer probe files from disk
        super().load_encoder_probes(epoch)

        # Get the number of tokens of the sentence
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = self. __get_sentence_tokens(sentence_id)

        # Get the input sentence words corresponding to the tokens
        input_words = list()
        for token in src_sentence_tokens:
            input_words.append(super().get_src_word_from_token(token))

        # Get the attention scores for all the heads of the specified attention layer
        attention_scores = self.__get_attention_scores(sentence_id, N_src_tokens, attention_layer)

        # Plot the attention scores
        plot_title = f"Attention Scores of encoder layer {attention_layer}"
        self.__plot_attention_scores(attention_scores, input_words, plot_title)


    def __process_probes(self):
        print(f"Running test: {self._test_id}")
        
        if self._test_id is None:
            self._test_id = 1


        match self._test_id:
            case 1:
                # Compute the column entropy and mutual information of the Query, Key and Value arrays
                # of the multihead attention layers
                self.__process_QKV_matrix()

            case 2:
                # Compute the column entropy and mutual information of the Query, Key and Value heads
                # of the multihead attention layers
                self.__process_QKV_heads()

            case 3:
                self.__process_attention_scores()

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

    model_config["tokenizer_dir"] = "../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../model_data/opus_books_en_fr/probes_8"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Analyze the Transformer's encoder emebdding layer probes
    analyzer = MultiheadAttentionAnalyzer(model_config, probe_config)
    analyzer.run(test=args.test)


# Entry point of the program
if __name__ == '__main__':
    main()