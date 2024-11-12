# Implementation of the configuration classes of the Model

# Base model configuration class
class ModelConfig:
    def __init__(self) -> None:
        super().__init__()
        self._config = dict()

    def get_config(self) -> dict:
        return self._config

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
            "tokenizer_dir" : "tokenizers/opus_books_en_fr",
            "model_dir"     : "models/opus_books_en_fr",
            "test_name": "runs/opus_books_en_fr"
        }

    # get configuration public method
    def get_config(self) -> dict:
        return super().get_config()

    