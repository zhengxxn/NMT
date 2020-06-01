import torch
import numpy as np
from util.decoding.beam_search import BeamSearch
from util.batch.transformer_batch import SrcTestBatch


def greedy_search(model, src, sos_index, eos_index, pad_index, max_len,):
    """Greedily decode a sentence."""

    batch = SrcTestBatch(src, pad_index)
    batch_size = src.size(0)

    with torch.no_grad():
        state = model.prepare_for_decode(batch, test=True)
        prev_y = torch.ones(batch_size).fill_(sos_index).type_as(src)  # sos

        predictions = []
        for i in range(max_len):
            log_probs, state = model.decode_step(prev_y, state)

            _, next_word = torch.max(log_probs, dim=1)
            prev_y = next_word.clone().type_as(src)
            predictions.append(prev_y.unsqueeze(1))

    predictions = torch.cat(predictions, dim=1)
    return {'prediction': predictions}
    # np.concatenate(predictions, axis=1)


def beam_search(model,
                src,
                sos_index, eos_index, pad_index,
                max_len, beam_size, length_penalty=False, alpha=-1.0,
                use_multiple_gpu=False):

    batch = SrcTestBatch(src, pad_index)
    batch_size = src.size(0)
    src_max_len = src.size(1)

    # do some initialization
    if use_multiple_gpu:
        initial_state = model.module.prepare_for_decode(batch.src, batch.src_mask)
    else:
        initial_state = model.prepare_for_decode(batch.src, batch.src_mask)

    prev_y = torch.ones(batch_size).fill_(sos_index).type_as(src)  # sos
    beam = BeamSearch(end_index=eos_index,
                      max_steps=max_len,
                      beam_size=beam_size,
                      length_penalty=length_penalty,
                      alpha=alpha)
    top_k_predictions, log_probabilities = beam.search(start_predictions=prev_y,
                                                       start_state=initial_state,
                                                       step=model.module.decode_step if use_multiple_gpu else model.decode_step)

    # decoding_states = model.get_state()

    return {'prediction': top_k_predictions[:, 0, :]}
