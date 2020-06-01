from torchtext.data.iterator import Iterator, BucketIterator
from util.data_loader.iterator.token_level_bucket_iterator import TokenLevelBucketIterator, max_token_batch_size_fn


def get_train_iterator(dataset, batch_size, device, token_level=True):
    if token_level:
        it = TokenLevelBucketIterator(dataset=dataset,
                                      batch_size=batch_size,
                                      device=device,
                                      repeat=True,
                                      sort_key=lambda x: (len(x.src), len(x.trg)),
                                      sort=False,
                                      sort_within_batch=True,
                                      batch_size_fn=max_token_batch_size_fn,
                                      train=True,
                                      shuffle=True)
    else:
        it = BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=lambda x: (len(x.src), len(x.trg)),
            train=True,
            shuffle=True,
            sort_within_batch=True,
            sort=False,
            device=device
        )
    return it


def get_dev_iterator(dataset, batch_size, device):
    it = Iterator(
        dataset=dataset,
        batch_size=batch_size,
        sort_key=lambda x: (len(x.src)),
        sort=False,
        shuffle=False,
        sort_within_batch=True,
        train=False,
        device=device
    )
    return it


def get_test_iterator(dataset, batch_size, device):
    it = Iterator(
        dataset=dataset,
        batch_size=batch_size,
        sort=False,
        shuffle=False,
        train=False,
        sort_within_batch=False,
        device=device
    )
    return it
