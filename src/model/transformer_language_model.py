import torch.nn as nn


class TransformerLanguageModel(nn.Module):
    """
    This Module implements the Transformer-type language model, includes an embedding layer, decoder(doest not have
    cross-attention layer), generator,
    """

    def __init__(self,
                 embedding_layer,
                 decoder,
                 generator,
                 vocab,
                 share_embedding=False,):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.decoder = decoder
        self.generator = generator
        self.vocab = vocab

        if share_embedding:
            self.generator.proj.weight = self.embedding_layer.embedding_layer.weight

    def forward(self, inp, mask):

        decoder_logits = self.decode(inp, mask)
        log_probs = self.generator(decoder_logits)

        return {'log_prob': log_probs}

    def decode(self, inp, mask):

        input_embedding = self.embedding_layer(inp)
        state = self.decoder(input_embedding, mask)
        return state['memory']
