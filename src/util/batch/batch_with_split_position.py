import torch
import numpy as np
from util.batch.transformer_batch import make_std_mask


class TrainingBatch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src_subword_mask, trg_subword_mask, batch, pad=0):
        self.src = batch.src
        trg = batch.trg

        self.src_mask = (self.src != pad).unsqueeze(-2).to(self.src.device)
        # if batch. is not None:

        self.trg_input = trg[:, :-1]
        self.trg = trg[:, 1:]

        self.trg_mask = make_std_mask(self.trg_input, pad)
        self.ntokens = (self.trg != pad).data.sum().to(self.trg.device)

        self.src_inner_index = torch.zeros_like(self.src, requires_grad=False).type_as(self.src).to(self.src.device)
        self.src_outer_index = torch.zeros_like(self.src, requires_grad=False).type_as(self.src).to(self.src.device)
        self.trg_inner_index = torch.zeros_like(self.trg, requires_grad=False).type_as(self.trg).to(self.src.device)
        self.trg_outer_index = torch.zeros_like(self.trg, requires_grad=False).type_as(self.trg).to(self.src.device)

        src_len = self.src.size(1)
        trg_input_len = self.trg_input.size(1)

        # create_for src:
        current_input = self.src[:, 0]
        last_is_sub_word = torch.gather(src_subword_mask, 0, current_input)
        last_inner_position_index = torch.zeros_like(current_input, requires_grad=False).type_as(current_input).to(self.src.device)
        last_outer_position_index = torch.zeros_like(current_input, requires_grad=False).type_as(current_input).to(self.src.device)
        self.src_inner_index[:, 0] = last_inner_position_index
        self.src_outer_index[:, 0] = last_outer_position_index
        for i in range(1, src_len):
            # [batch size, 1]
            current_input = self.src[:, i]
            current_inner_position = torch.where(last_is_sub_word == 1,
                                                 last_inner_position_index + 1,
                                                 torch.zeros_like(current_input).type_as(current_input).to(self.src.device))
            current_outer_position = torch.where(last_is_sub_word == 1,
                                                 last_outer_position_index,
                                                 last_outer_position_index + 1)
            self.src_inner_index[:, i] = current_inner_position
            self.src_outer_index[:, i] = current_outer_position

            # update last state
            last_inner_position_index = current_inner_position
            last_outer_position_index = current_outer_position
            last_is_sub_word = torch.gather(src_subword_mask, 0, current_input)

        current_input = self.trg_input[:, 0]
        last_is_sub_word = torch.gather(trg_subword_mask, 0, current_input)
        last_inner_position_index = torch.zeros_like(current_input).type_as(current_input)
        last_outer_position_index = torch.zeros_like(current_input).type_as(current_input)
        self.trg_inner_index[:, 0] = last_inner_position_index
        self.trg_outer_index[:, 0] = last_outer_position_index
        for i in range(1, trg_input_len):
            # [batch size, 1]
            current_input = self.trg_input[:, i]
            current_inner_position = torch.where(last_is_sub_word == 1,
                                                 last_inner_position_index + 1,
                                                 torch.zeros_like(current_input).type_as(current_input).to(self.src.device))
            current_outer_position = torch.where(last_is_sub_word == 1,
                                                 last_outer_position_index,
                                                 last_outer_position_index + 1)
            self.trg_inner_index[:, i] = current_inner_position
            self.trg_outer_index[:, i] = current_outer_position

            # update last state
            last_inner_position_index = current_inner_position
            last_outer_position_index = current_outer_position
            last_is_sub_word = torch.gather(trg_subword_mask, 0, current_input)

        # self.src_inner_index = batch.src_inner_index
        # self.src_outer_index = batch.src_outer_index
        # self.trg_inner_index = batch.trg_inner_index
        # self.trg_outer_index = batch.trg_outer_index


class SrcTestBatch:
    def __init__(self, src, pad, src_subword_mask):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        self.src_inner_index = torch.zeros_like(self.src, requires_grad=False).type_as(self.src).to(self.src.device)
        self.src_outer_index = torch.zeros_like(self.src, requires_grad=False).type_as(self.src).to(self.src.device)

        src_len = self.src.size(1)
        current_input = self.src[:, 0]
        last_is_sub_word = torch.gather(src_subword_mask, 0, current_input)
        last_inner_position_index = torch.zeros_like(current_input, requires_grad=False).type_as(current_input).to(self.src.device)
        last_outer_position_index = torch.zeros_like(current_input, requires_grad=False).type_as(current_input).to(self.src.device)
        self.src_inner_index[:, 0] = last_inner_position_index
        self.src_outer_index[:, 0] = last_outer_position_index
        for i in range(1, src_len):
            # [batch size, 1]
            current_input = self.src[:, i]
            current_inner_position = torch.where(last_is_sub_word == 1,
                                                 last_inner_position_index + 1,
                                                 torch.zeros_like(current_input).type_as(current_input).to(self.src.device))
            current_outer_position = torch.where(last_is_sub_word == 1,
                                                 last_outer_position_index,
                                                 last_outer_position_index + 1)
            self.src_inner_index[:, i] = current_inner_position
            self.src_outer_index[:, i] = current_outer_position

            # update last state
            last_inner_position_index = current_inner_position
            last_outer_position_index = current_outer_position
            last_is_sub_word = torch.gather(src_subword_mask, 0, current_input)

        # self.src_inner_index = src_inner_index
        # self.src_outer_index = src_outer_index


class TrgTestBatch:
    def __init__(self, trg_input, pad, trg_inner_index, trg_outer_index):
        self.trg_input = trg_input
        self.trg_mask = make_std_mask(trg_input, pad)

        self.trg_inner_index = trg_inner_index
        self.trg_outer_index = trg_outer_index

        #  subsequent_mask(trg_input.size(1)).to(self.trg_input.device)
