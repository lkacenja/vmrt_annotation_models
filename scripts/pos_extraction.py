import argparse
import glob
import os

import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag

"""
Script that atomizes a plane text file by sentence and then word, keeping only the proper nouns.

Example usage:

python scripts/pos_extraction.py data/*.txt ./data
"""

# The parts of speach we want to keep.
pos_of_interest = ('NNPS', 'NNP', 'NNS', 'NN')

def extract_pos(args: argparse.Namespace) -> None:
    """
    Performs part os speach extraction for all supplied files.

    Parameters
    ----------
    args: argparse.Namespace
      The parsed args.
    """
    result = glob.glob(args.input_file)
    if len(result) == 0:
        RuntimeError('No input files found!')
    for file_path in result:
        with open(file_path, 'r') as file:
            data = file.read()
            data = data.replace('\n', ' ')
            sentences = sent_tokenize(data)
        output = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            output += [{'word': x[0], 'pos': x[1]} for x in pos_tag(words)]
        output_df = pd.DataFrame(output)
        output_df = output_df.loc[output_df['pos'].isin(pos_of_interest), :]
        output_df = output_df.loc[output_df['word'].str.len() > 3, :]
        path_bits = os.path.split(file_path)
        file_bits = os.path.splitext(path_bits[1])
        output_df.to_csv(f'{args.output_file}/{file_bits[0]}_pos{file_bits[1]}', index=False)

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
    parser.add_argument('input_file', help='Path to input files allows splats.')
    parser.add_argument('output_file', help='Where to place the output files.')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    extract_pos(provided_args)
