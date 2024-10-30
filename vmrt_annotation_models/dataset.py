import torch

"""
A module providing our customized data source.
"""


class CategorizationDataset(torch.utils.data.Dataset):
    """
    A custom Dataset built to navigate our categorization data with: input and list keys.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input = self.data.input
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_text = str(self.input[index])
        input_text = ' '.join(input_text.split())

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
