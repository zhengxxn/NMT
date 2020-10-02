import torch
import numpy as np


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    _subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(_subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & torch.tensor(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


class TrainingBatch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, batch, pad=0):

        self.src = batch.src
        trg = batch.trg

        self.src_mask = (self.src != pad).unsqueeze(-2).to(self.src.device)
        # if batch. is not None:

        self.trg_input = trg[:, :-1]
        self.trg = trg[:, 1:]

        self.trg_input_mask = (self.trg_input != pad).unsqueeze(-2).to(self.src.device)
        self.trg_mask = make_std_mask(self.trg_input, pad).to(self.trg.device)
        self.ntokens = (self.trg != pad).data.sum().to(self.trg.device)


class SrcTestBatch:

    def __init__(self, src, pad):

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2).to(src.device)


class TrgTestBatch:
    def __init__(self, trg_input, pad):
        self.trg_input = trg_input
        self.trg_mask = make_std_mask(trg_input, pad).to(trg_input.device)
        #  subsequent_mask(trg_input.size(1)).to(self.trg_input.device)
