import argparse
import os

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer

from vmrt_annotation_models.config import Config

"""
A script for performing multilabel text classification inference.

Example usage:

python scripts/inference_categorizer.py models/categorizer.bin data/category_output.csv data/pos_output.csv data
"""


def inference(device: str, model: torch.nn.Module, tokenizer: BertTokenizer, labels: list) -> callable:
    """
    Returns the main inference function.

    Parameters
    ----------
    device: str
      The device type.
    model: torch.nn.Module
      Our model instance.
    training_loader: torch.utils.data.DataLoader
      The training data loader.
    labels: list
      The labels used for returning human-readable results (used in training).

    Returns
    -------
    The inference function.
    """

    def _inference(text: str):
        """
        Performs multi label text classification on the provided string.

        Parameters
        ----------
        text: str
          The text to evaluate.

        Returns
        -------
        A pandas series with confidence and type indexes.
        """
        # Encode the text and prepare inputs for the model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = inputs['input_ids'].to(device, dtype=torch.long)
        mask = inputs['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long)

        fin_outputs = []
        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        outputs = np.array(fin_outputs)
        sorted_probs = sorted(((prob.item(), labels[idx]) for idx, prob in enumerate(outputs[0])), reverse=True,
                              key=lambda x: x[0])
        best_result = sorted_probs[0]
        return pd.Series(best_result)

    return _inference


def do_infer(args: argparse.Namespace) -> None:
    """
    An entry function to run the whole inference process.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    """
    dict_df = pd.read_csv(args.training_data, low_memory=False)
    dict_df = pd.get_dummies(dict_df, columns=['type'], dtype=int)
    labels = dict_df.columns[1:].to_list()
    input = pd.read_csv(args.input_data, low_memory=False)
    device = Config.get_device()
    model = Config.get_model()
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    tokenizer = Config.get_tokenizer()
    inference_fn = inference(device, model, tokenizer, labels)
    input[['confidence', 'type']] = input['word'].apply(inference_fn)
    path_bits = os.path.split(args.input_data)
    file_bits = os.path.splitext(path_bits[1])
    input.to_csv(f'{args.output_path}/{file_bits[0]}_annotated{file_bits[1]}', index=False)


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Removes unresolved columns from dictionaries.')
    parser.add_argument('model_path', help='Where to find the pretrained model.')
    parser.add_argument('training_data', help='Training data to extract labels from.')
    parser.add_argument('input_data', help='Data to perform inference on.')
    parser.add_argument('output_path', help='Where to place the file with our inference results.')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    do_infer(provided_args)
