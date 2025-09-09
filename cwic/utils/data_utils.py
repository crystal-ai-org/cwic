import torch

import numpy as np


class TokenCollator:

    def __init__(self, tokenizer, max_length, device=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    
    def __call__(self, batch):

        texts = [b["text"] for b in batch]
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            padding_side="right",
        ).input_ids

        out = {
            "input_ids": input_ids.to(self.device),
        }

        return out


class DeviceCollator:

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def __call__(self, batch):

        out = []
        for b in batch:

            curr_out = {}
            for k, v in b.items():

                if isinstance(v, torch.Tensor):
                    curr_out[k] = v.to(self.device)

                elif isinstance(v, np.ndarray):
                    curr_out[k] = torch.from_numpy(v).to(self.device)

                elif isinstance(v, list):
                    curr_out[k] = torch.tensor(v).to(self.device)

            out.append(curr_out)

        out_dict = {}
        for k in out[0].keys():
            out_dict[k] = torch.stack([b[k] for b in out])

        return out_dict
