import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader

from vmrt_annotation_models.dataset import CategorizationDataset
from vmrt_annotation_models.config import Config

"""
A script for training a BERT model in multilabel text classification.

Example usage:

python scripts/train_categorizer.py  data/category_output.csv ./models
"""


def process_data(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares input data for training purposes.

    Parameters
    ----------
    training_data: pd.DataFrame
      A preloaded dataframe with our dictionary contents.

    Returns
    -------
    A dataframe formatted for training.
    """
    training_data = training_data.dropna()
    training_data = pd.get_dummies(training_data, columns=['type'], dtype=int)
    print('Total data columns: ', len(training_data.columns) - 1)
    training_data['list'] = training_data[training_data.columns[1:]].values.tolist()
    training_data = training_data[['input', 'list']]
    return training_data


def get_data_loaders(training_data: pd.DataFrame, training_split) -> tuple:
    """
    Breaks training data into training and validation data sets and returns appropriate DataLoaders.

    Parameters
    ----------
    training_data: pd.DataFrame
      The training dataset.
    training_split: float
      The proportion of the dataset to include in the training set.

    Returns
    -------
    A tuple of training and validation data loaders.
    """
    train_dataset = training_data.sample(frac=training_split, random_state=200)
    test_dataset = training_data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(training_data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = Config.get_tokenizer()
    training_set = CategorizationDataset(train_dataset, tokenizer, Config.max_len)
    testing_set = CategorizationDataset(test_dataset, tokenizer, Config.max_len)

    train_params = {'batch_size': Config.train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': Config.valid_batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def train(device: str, model: torch.nn.Module, training_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer) -> float:
    """
    Performs one epoch of training.

    Parameters
    ----------
    device: str
      The device type.
    model: torch.nn.Module
      Our model instance.
    training_loader: torch.utils.data.DataLoader
      The training data loader.
    optimizer: torch.optim.Optimizer
      The optimizer instance to user.

    Returns
    -------
    The average loss for the epoch.
    """
    model.train()
    loss_ = 0
    for data in tqdm(training_loader, total=len(training_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
        loss_ += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_ / len(training_loader)


def validation(device: str, model: torch.nn.Module, testing_loader: torch.utils.data.DataLoader) -> tuple:
    """
    Validates a trained model.

    Parameters
    ----------
    device: str
      The device type.
    model: torch.nn.Module
      Our model instance.
    testing_loader: torch.utils.data.DataLoader
      The validation (testing) data loader.

    Returns
    -------
    Tuple with model predictions and actual values.
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def do_train_model(args: argparse.Namespace) -> None:
    """
    An entry function to run the whole training process.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    """
    model = Config.get_model()
    device = Config.get_device()
    model.to(device)
    training_data = pd.read_csv(args.training_data, low_memory=False)
    training_data = process_data(training_data)
    training_loader, testing_loader = get_data_loaders(training_data, Config.training_split)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.learning_rate)
    for epoch in range(Config.epochs):
        print(f'Beginning epoch {epoch}')
        loss_item = train(device, model, training_loader, optimizer)
        print(f'Epoch: {epoch}, Loss: {loss_item}')
    for epoch in range(Config.epochs):
        outputs, targets = validation(device, model, testing_loader)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
    torch.save(model.state_dict(), f'{args.model_path}/categorizer.bin')


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Trains a BERT model for multilabel text classification.')
    parser.add_argument('training_data', help='Path to csv with training data.')
    parser.add_argument('model_path', help='Where to place the model.')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    do_train_model(provided_args)
