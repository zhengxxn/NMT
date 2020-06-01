import torch
import numpy as np
from util.decoding.beam_search import BeamSearch
from util.batch.batch_with_split_position import SrcTestBatch


def beam_search(model,
                src, src_subword_mask,
                sos_index, eos_index, pad_index,
                max_len, beam_size, length_penalty=False, alpha=-1.0,
                use_multiple_gpu=False):

    # batch = SrcTestBatch(src, pad_index, src_inner_index=src_inner_index, src_outer_index=src_outer_index)
    batch = SrcTestBatch(src=src, pad=pad_index, src_subword_mask=src_subword_mask)
    batch_size = src.size(0)

    # do some initialization
    if use_multiple_gpu:
        initial_state = model.module.prepare_for_decode(batch.src, batch.src_mask,
                                                        batch.src_inner_index, batch.src_outer_index)
    else:
        initial_state = model.prepare_for_decode(batch.src, batch.src_mask,
                                                 batch.src_inner_index, batch.src_outer_index)

    prev_y = torch.ones(batch_size).fill_(sos_index).type_as(src)  # sos
    beam = BeamSearch(end_index=eos_index,
                      max_steps=max_len,
                      beam_size=beam_size,
                      length_penalty=length_penalty,
                      alpha=alpha)
    top_k_predictions, log_probabilities = beam.search(start_predictions=prev_y,
                                                       start_state=initial_state,
                                                       step=model.module.decode_step if use_multiple_gpu else model.decode_step)

    return {'prediction': top_k_predictions[:, 0, :]}
