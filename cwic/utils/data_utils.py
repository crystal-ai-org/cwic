import torch
import torch.nn.functional as F

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


class PackedCollator:

    def __init__(self, tokenizer, max_length, device=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    
    def __call__(self, batch):

        input_ids = [torch.tensor(b["input_ids"]).long() for b in batch]
        segment_ids = [torch.tensor(b["segment_ids"]).long() for b in batch]

        # pad into single tensor
        out = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        seg_out = torch.nn.utils.rnn.pad_sequence(
            segment_ids,
            batch_first=True,
            padding_value=-1
        )

        # apply seq_length constraint
        if out.shape[1] < self.max_length:
            out = F.pad(
                out,
                (0, self.max_length - out.shape[1]),
                value=self.tokenizer.pad_token_id
            )
        elif out.shape[1] > self.max_length:
            out = out[:, :self.max_length]

        if seg_out.shape[1] < self.max_length:
            seg_out = F.pad(
                seg_out,
                (0, self.max_length - seg_out.shape[1]),
                value=-1
            )
        elif seg_out.shape[1] > self.max_length:
            seg_out = seg_out[:, :self.max_length]
        
        input_ids = out.to(self.device)
        segment_ids = seg_out.to(self.device)

        # create 4D attention mask
        attention_mask = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]

        # add causal mask
        position = torch.arange(input_ids.shape[1], device=input_ids.device)
        causal = position[None, None, :, None] >= position[None, None, None, :]
        attention_mask = attention_mask & causal

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
