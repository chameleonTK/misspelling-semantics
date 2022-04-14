
import glob
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
# from thai2transformers.datasets import SequenceClassificationDataset
import torch

class SequenceClassificationDataset(TorchDataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=128,
        ext=".csv",
        bs=10000,
        preprocessor=None,
        input_ids=[],
        misp_ids=[],
        attention_masks=[],
        labels=[],
        label_encoder=None
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.preprocessor = preprocessor
        self.input_ids = input_ids
        self.misp_ids = misp_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.label_encoder = label_encoder
        # self._build()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "misp_ids": torch.tensor(self.misp_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[i], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }

