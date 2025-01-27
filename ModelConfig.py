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
            "datasource": 'opus_books',
            "lang_src": "en",
            "lang_tgt": "fr",
            "preload": "latest",
            "tokenizer_dir" : "model_data/opus_books_en_fr/tokens",
            "model_dir"     : "model_data/opus_books_en_fr/weights",
            "test_name": "model_data/opus_books_en_fr/runs",
            "dataset_dir": "model_data/opus_books_en_fr/dataset",
            "probe_dir": "model_data/opus_books_en_fr/probes",
        }

        self._probes = {
            "enc_embed_layer" : "enc_embedding_probe",
            "enc_layer_0_attn": "enc0_multi_head_probe",
            "enc_layer_0_feedforward": "enc0_feedforward_probe",
            "enc_layer_5_attn": "enc5_multi_head_probe",
            "enc_layer_5_feedforward": "enc5_feedforward_probe",
            "enc_block_probe": "encoder_probe"
        }

    # get configuration public method
    def get_config(self) -> dict:
        return super().get_config()

    # get probes public method
    def get_probes(self) -> dict:
        return super().get_probes()

    