from .transformer_batch import subsequent_mask, make_std_mask


class TrainingBatch:
    def __init__(self, batch, pad=0):

        text = batch.text
        # if batch. is not None:

        self.inp = text[:, :-1]
        self.trg = text[:, 1:]

        self.trg_mask = make_std_mask(self.inp, pad).to(self.trg.device)
        self.ntokens = (self.trg != pad).data.sum().to(self.trg.device)
