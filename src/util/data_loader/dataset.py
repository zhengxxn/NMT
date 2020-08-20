from torchtext.data import TabularDataset, Dataset, Example


def filter_dataset(dataset: TabularDataset, data_fields, filter_len=None):
    examples = dataset.examples
    new_dataset = load_dataset_from_example(examples=examples, data_fields=data_fields, max_len=filter_len)
    return new_dataset


def load_datasets(paths: list, data_fields, filter_len=None):
    datasets = [filter_dataset(
        dataset=TabularDataset(
            path=p,
            format='tsv',
            fields=data_fields,
            skip_header=True
        ),
        # data_fields={'src': data_fields['src'][1], 'trg': data_fields['trg'][1]},
        data_fields=data_fields,
        filter_len=filter_len)
        for p in paths]

    return datasets


def combine_datasets(datasets: list, data_fields, filter_len=None):
    examples = datasets[0].examples
    for i in range(1, len(datasets)):
        examples += datasets[i].examples
    dataset = load_dataset_from_example(examples=examples,
                                        data_fields=data_fields,
                                        max_len=filter_len)
    return dataset


def load_dataset_from_example(examples: Example, data_fields, max_len=None):
    if max_len is not None:
        dataset = Dataset(examples, data_fields, filter_pred=
        lambda x: len(x.src) <= max_len and len(x.trg) <= max_len)
        # lambda x: len(vars(x)['src']) <= max_len and len(vars(x)['trg']) <= max_len)
    else:
        dataset = Dataset(examples, data_fields)

    return dataset