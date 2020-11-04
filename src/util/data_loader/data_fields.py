from torchtext.data import Field
import torch


def mt_data_wo_length_fields():
    src_text = Field(sequential=True,
                     use_vocab=True,
                     batch_first=True,
                     # init_token='<sos>',
                     # eos_token='<eos>',
                     include_lengths=False)

    trg_text = Field(sequential=True,
                     use_vocab=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     batch_first=True,
                     include_lengths=False,
                     is_target=True)

    data_fields = [('src', src_text), ('trg', trg_text)]
    # data_fields = {'src': ('src', src_text), 'trg': ('trg', trg_text)}
    return data_fields


def lm_data_fields():
    """
    Language Model Data Field, Only contains Text
    """

    text = Field(sequential=True,
                 use_vocab=True,
                 batch_first=True,
                 init_token='<sos>',
                 eos_token='<eos>',
                 include_lengths=False,
                 is_target=True)

    data_field = [('text', text)]
    return data_field


def mt_data_fields():
    src_text = Field(sequential=True,
                     use_vocab=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     batch_first=True,
                     include_lengths=True)

    trg_text = Field(sequential=True,
                     use_vocab=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     batch_first=True,
                     include_lengths=True,
                     is_target=True)

    data_fields = [('src', src_text), ('trg', trg_text)]
    # data_fields = {'src': ('src', src_text), 'trg': ('trg', trg_text)}
    return data_fields


def mt_with_sent_label_data_fields():
    src_text = Field(sequential=True,
                     use_vocab=True,
                     batch_first=True,
                     include_lengths=True)

    trg_text = Field(sequential=True,
                     use_vocab=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     batch_first=True,
                     include_lengths=True,
                     is_target=True)

    sent_label = Field(sequential=True,
                       use_vocab=True,
                       batch_first=True,
                       include_lengths=True,
                       unk_token=None,
                       )

    data_fields = [('src', src_text), ('trg', trg_text), ('src_label', sent_label), ('trg_label', sent_label)]

    return data_fields


def mt_with_split_position_data_fields():
    src_text = Field(sequential=True,
                     use_vocab=True,
                     batch_first=True,
                     include_lengths=False)

    trg_text = Field(sequential=True,
                     use_vocab=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     batch_first=True,
                     include_lengths=False,
                     is_target=True)

    def numerical_tokenize(x):
        x = x.split(' ')
        x = [int(v) for v in x]
        return x

    numerical_field = Field(
        sequential=True,
        use_vocab=False,
        dtype=torch.long,
        pad_token=0,
        unk_token=None,
        batch_first=True,
        tokenize=numerical_tokenize,
        include_lengths=False,
    )

    # sent_label = Field(sequential=True,
    #                    use_vocab=True,
    #                    batch_first=True,
    #                    include_lengths=True)

    data_fields = [('src', src_text), ('trg', trg_text),
                   ('src_outer_index', numerical_field), ('src_inner_index', numerical_field),
                   ('trg_outer_index', numerical_field), ('trg_inner_index', numerical_field)]

    return data_fields
