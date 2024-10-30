# vmrt_annotation_models
Various attempts at automated medical record annotation. Since the corpus of Golden Retriever Lifetime Study data does not include significant label data for named entity recognition, a good approach for annotation, we have to improvise and work with what we have. Here is the rough process:
1. Reduce textual dataset (EMR) to just proper nouns. This helps to remove clutter/noise.
2. Assemble study data from dictionaries into categorical dataset. This provides human input paired with low effort labeling.
3. Train a BERT model for multilabel text classification with the categorical dataset.
4. Perform inference on input proper noun list.

The output gives semi-reliable prediction of categories within the textual dataset.

## Environment
At the moment, the code in this repo is meant for local execution only. The Morris Animal Foundation Data Science Team work on Apple M3 or later machines. The easiest way we found to access the GPUs on these machines is to use venv and mpu. Unfortunately, this prevents containerization for local training. Our steps to set up are as follows:
1. Initialize the venv: `python3 -m venv .venv`
2. Change `.venv/bin` permissions as needed.
3. Run `.venv/bin/activate`
4. Run `.venv/bin/pip install -r requirements.txt`
5. Execute scripts with `.venv/bin/python scripts/<script>.py`

## Process
Roughly the scripts are intended to be run in the following order:
1. `train_categorizor.py` to train the model.
2. `pos_extraction.py` on EMR textual resources.
3. `inference_categorizor.py` to provide categorization of data extracted in #2.