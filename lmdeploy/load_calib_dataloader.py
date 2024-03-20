# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/utils/calib_dataloader.py


def get_ptb():
    """Load PTB train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')


if __name__ == '__main__':
    get_ptb()


