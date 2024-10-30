import torch
from transformers import BertTokenizer
from vmrt_annotation_models.model import BERTModel

"""
Stores a default configuration object for training and inference with our BERT model.
"""


class Config:
    """
    Hyperparemters and other environmental configuration options.
    """
    model_name = 'havocy28/VetBERT'
    max_len = 200
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 1
    learning_rate = 1e-05
    training_split = 0.8

    @staticmethod
    def get_tokenizer():
        return BertTokenizer.from_pretrained(Config.model_name)

    @staticmethod
    def get_model():
        return BERTModel(Config.model_name, 3)

    @staticmethod
    def get_device():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Device is {device}... prepare for power!')
        return device