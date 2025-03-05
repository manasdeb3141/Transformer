# Implementation of the configuration classes of the Model

# Base model configuration class
class ModelConfig:
    def __init__(self) -> None:
        super().__init__()
        self._config = dict()
        self._probes = dict()

    def get_config(self) -> dict:
        return self._config

    def get_probes(self) -> dict:
        return self._probes

# Language model configuration class inherits ModelConfig class
class LangModelConfig(ModelConfig):
    # Constructor
    def __init__(self) -> None:
        super().__init__()

        self._config = {
            "batch_size": 8,
            "num_epochs": 20,
            "lr": 10**-4,
            "seq_len": 350,
            "d_model": 512,
            "train_probe_count" : 100,
            "datasource": 'opus_books',
            "lang_src": "en",
            "lang_tgt": "fr",
            "preload": "latest",
            "tokenizer_dir" : "model_data/opus_books_en_fr/tokens",
            "model_dir"     : "model_data/opus_books_en_fr/weights",
            "test_name": "model_data/opus_books_en_fr/runs",
            # "dataset_dir": "model_data/opus_books_en_fr/dataset",
            "probe_dir": "model_data/opus_books_en_fr/probes",
            "train_probe_dir" : "model_data/opus_books_en_fr/train_probes",
            "analyze_dir": "model_data/opus_books_en_fr/probes",
            "use_special_dataset" : 0,
            "only_validate": True
        }

        self._probes = {
            # Encoder
            "enc_embed_layer" : "enc_embedding_probe",
            "enc_layer_0_attn": "enc0_self_attn_probe",
            "enc_layer_0_feedforward": "enc0_feedforward_probe",
            "enc_layer_1_attn": "enc1_self_attn_probe",
            "enc_layer_1_feedforward": "enc1_feedforward_probe",
            "enc_layer_2_attn": "enc2_self_attn_probe",
            "enc_layer_2_feedforward": "enc2_feedforward_probe",
            "enc_layer_3_attn": "enc3_self_attn_probe",
            "enc_layer_3_feedforward": "enc3_feedforward_probe",
            "enc_layer_4_attn": "enc4_self_attn_probe",
            "enc_layer_4_feedforward": "enc4_feedforward_probe",
            "enc_layer_5_attn": "enc5_self_attn_probe",
            "enc_layer_5_feedforward": "enc5_feedforward_probe",
            "enc_block": "encoder_probe",

            # Decoder
            "dec_embed_layer" : "dec_embedding_probe",
            "dec_layer_0_attn": "dec0_self_attn_probe",
            "dec_layer_0_cross_attn": "dec0_cross_attn_probe",
            "dec_layer_0_feedforward": "dec0_feedforward_probe",
            "dec_layer_1_attn": "dec1_self_attn_probe",
            "dec_layer_1_cross_attn": "dec1_cross_attn_probe",
            "dec_layer_1_feedforward": "dec1_feedforward_probe",
            "dec_layer_2_attn": "dec2_self_attn_probe",
            "dec_layer_2_cross_attn": "dec2_cross_attn_probe",
            "dec_layer_2_feedforward": "dec2_feedforward_probe",
            "dec_layer_3_attn": "dec3_self_attn_probe",
            "dec_layer_3_cross_attn": "dec3_cross_attn_probe",
            "dec_layer_3_feedforward": "dec3_feedforward_probe",
            "dec_layer_4_attn": "dec4_self_attn_probe",
            "dec_layer_4_cross_attn": "dec4_cross_attn_probe",
            "dec_layer_4_feedforward": "dec4_feedforward_probe",
            "dec_layer_5_attn": "dec5_self_attn_probe",
            "dec_layer_5_cross_attn": "dec5_cross_attn_probe",
            "dec_layer_5_feedforward": "dec5_feedforward_probe",
            "dec_block": "decoder_probe",
            
            # Projection Layer
            "proj_layer": "projection_probe"
        }

    # get configuration public method
    def get_config(self) -> dict:
        return super().get_config()

    # get probes public method
    def get_probes(self) -> dict:
        return super().get_probes()

    