import torch
from transformers import BertConfig, BertModel, BertTokenizer

"""
A module including our model for training and inference purposes.
"""


class BERTModel(torch.nn.Module):
    """
    Our model for multilabel text classification.
    """

    def __init__(self, model_name, layers):
        super(BERTModel, self).__init__()
        self.l1 = BertModel.from_pretrained(model_name)
        if self.training:
            print('Model is in training mode, including dropout layer.')
            self.l2 = torch.nn.Dropout(0.3)
        else:
            print('Model is in inference mode, excluding dropout layer.')
        self.l3 = torch.nn.Linear(768, layers)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        if self.training:
            output = self.l2(output)
        output = self.l3(output)
        return output
