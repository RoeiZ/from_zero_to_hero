"""Synthetic sequence-classification dataset: "majority token".

Task: given a sequence of L integer tokens drawn from a small vocabulary
of size V, predict which token value is the (unique) majority element
of the sequence.

Why this task for a generic Transformer encoder (not yet patch/image
based): classifying the majority element genuinely requires aggregating
information from *every* position in the sequence -- a model that only
looks at a fixed position (e.g. position 0) or local neighbors cannot
solve it in general, because the majority value can sit at any of the L
positions and is only defined by counting across the whole sequence.
This makes self-attention's "look at everything, weigh by relevance"
mechanism directly useful and directly visualizable (Sec. further down:
code/transformers/majority_token_evaluate.py plots how much the CLS
token attends to each position, and majority-value positions should
stand out).

Construction (guarantees the label is always the unique, unambiguous
mode of the sequence, so there is never a labeling ambiguity to debug):
    1. Sample majority_value ~ Uniform{0, ..., V-1}.
    2. k = L // 2 + 1 positions (a strict majority of the L positions,
       chosen by a random permutation so the majority value is not tied
       to a fixed position) are set to majority_value.
    3. The remaining L - k positions are filled with i.i.d.
       Uniform{0, ..., V-1} tokens (may coincidentally also equal
       majority_value, which only strengthens the majority).
    Because k > L - k (k = floor(L/2) + 1 is a strict majority of L),
    no other single value can appear k or more times among the
    remaining L - k < k random slots, so majority_value is guaranteed
    to be the unique mode.

Symbol -> code variable -> shape -> meaning:
    L (sequence length)    seq_len         scalar      number of tokens per sequence
    V (vocabulary size)    vocab_size      scalar      number of distinct token values (= number of classes)
    sequence                sequences[i]    [L]         int64 token ids, values in [0, V)
    majority_value           labels[i]       scalar      int64 class label, in [0, V)

Data split: train and test sequences are generated independently with
different seeds (seed and seed + 1), so there is no accidental overlap
between the two splits; both splits use the exact same generative
process (same vocab_size, seq_len).

Local usage (lightweight, CPU, tiny sizes, no GPU):
    python code/transformers/majority_token_dataset.py --smoke-test
"""

import argparse

import torch
from torch.utils.data import Dataset


class MajorityTokenDataset(Dataset):
    """Fixed, pre-generated dataset of (sequence, majority_label) pairs."""

    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            sequences: [num_samples, seq_len] int64
            labels:    [num_samples] int64
        """
        assert sequences.shape[0] == labels.shape[0]
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]  # [seq_len], scalar


def _generate_majority_token_sequences(
    num_samples: int, vocab_size: int, seq_len: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates `num_samples` sequences with a guaranteed unique majority
    token each, as described in the module docstring.

    Returns:
        sequences: [num_samples, seq_len] int64, values in [0, vocab_size)
        labels:    [num_samples] int64, values in [0, vocab_size)
    """
    assert seq_len >= 3, "seq_len must be >=3 for a strict majority to be a meaningful constraint"
    generator = torch.Generator().manual_seed(seed)

    num_majority_positions = seq_len // 2 + 1  # k: strict majority of seq_len positions

    labels = torch.randint(0, vocab_size, (num_samples,), generator=generator)  # [num_samples]

    # Fill every position with an i.i.d. random token first: [num_samples, seq_len].
    sequences = torch.randint(0, vocab_size, (num_samples, seq_len), generator=generator)

    # For each sample, pick k distinct positions (via a random permutation of
    # 0..seq_len-1, keeping the first k) and overwrite them with that sample's
    # majority value -- this is what guarantees the strict majority.
    for sample_index in range(num_samples):
        permuted_positions = torch.randperm(seq_len, generator=generator)
        majority_positions = permuted_positions[:num_majority_positions]  # [k]
        sequences[sample_index, majority_positions] = labels[sample_index]

    return sequences, labels


def make_majority_token_datasets(
    vocab_size: int, seq_len: int, num_train: int, num_test: int, seed: int
) -> tuple[MajorityTokenDataset, MajorityTokenDataset]:
    """Builds the train and test splits. Train and test use different
    seeds (seed, seed+1) so the two splits are generated independently."""
    train_sequences, train_labels = _generate_majority_token_sequences(
        num_train, vocab_size, seq_len, seed=seed
    )
    test_sequences, test_labels = _generate_majority_token_sequences(
        num_test, vocab_size, seq_len, seed=seed + 1
    )
    train_dataset = MajorityTokenDataset(train_sequences, train_labels)
    test_dataset = MajorityTokenDataset(test_sequences, test_labels)
    return train_dataset, test_dataset


def _smoke_test():
    """Lightweight local checks: shapes, value range, and -- critically --
    that every generated label really is the unique mode of its sequence
    (recomputed independently via torch.bincount, not trusted blindly)."""
    vocab_size, seq_len, num_train, num_test = 7, 11, 64, 32
    train_dataset, test_dataset = make_majority_token_datasets(
        vocab_size, seq_len, num_train, num_test, seed=0
    )

    assert len(train_dataset) == num_train
    assert len(test_dataset) == num_test
    assert train_dataset.sequences.shape == (num_train, seq_len)
    assert train_dataset.labels.shape == (num_train,)

    for dataset in (train_dataset, test_dataset):
        assert dataset.sequences.min() >= 0 and dataset.sequences.max() < vocab_size
        assert dataset.labels.min() >= 0 and dataset.labels.max() < vocab_size

        for sequence, label in zip(dataset.sequences, dataset.labels):
            counts = torch.bincount(sequence, minlength=vocab_size)  # [vocab_size]
            true_mode = counts.argmax()
            assert counts[label] > seq_len // 2, "label is not a strict majority"
            assert true_mode.item() == label.item(), "label is not the unique mode"

    # Train/test independence: generated with different seeds, so the raw
    # sequence tensors should not be byte-identical (extremely unlikely by
    # chance if generation is actually seed-dependent).
    assert not torch.equal(train_dataset.sequences[: len(test_dataset)], test_dataset.sequences)

    print("majority_token_dataset smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run lightweight local checks and exit")
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        train_dataset, test_dataset = make_majority_token_datasets(
            args.vocab_size, args.seq_len, args.num_train, args.num_test, args.seed
        )
        print(f"train: {len(train_dataset)} samples, sequences {tuple(train_dataset.sequences.shape)}")
        print(f"test:  {len(test_dataset)} samples, sequences {tuple(test_dataset.sequences.shape)}")
        for i in range(min(3, len(train_dataset))):
            sequence, label = train_dataset[i]
            print(f"  sequence={sequence.tolist()} majority_label={label.item()}")
