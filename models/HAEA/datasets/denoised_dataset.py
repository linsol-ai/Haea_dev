# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright Facebook, Inc. and its affiliates. Licensed under the MIT License as part of
# fairseq package.

import numpy as np
import torch
from torch.utils.data import Dataset
import math
import contextlib
from typing import List

class HaeaVocab:

    SPECIAL_TOKEN_PAD = 0
    SPECIAL_TOKEN_BOS = 1
    SPECIAL_TOKEN_EOS = 2
    SPECIAL_TOKEN_MASK = 3
    SPECIAL_TOKENS = [SPECIAL_TOKEN_PAD, SPECIAL_TOKEN_BOS, SPECIAL_TOKEN_EOS, SPECIAL_TOKEN_MASK]


    def __init__(self, source_dataset: torch.Tensor, src_var_list: List[str], label_dataset: torch.Tensor, label_var_list: List[str],
                  max_len, n_only_input: int = 0):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.label_dataset = label_dataset
        self.n_only_input = n_only_input
        self.pe = self.positional_encoding(source_dataset.size(-1), max_len + 10)
        self.max_len = max_len


    def __len__(self):
        return self.source_dataset.size(0) + len(self.SPECIAL_TOKENS)

    
    def positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


    def get_data(self, indicate, dataset: torch.Tensor, source: bool = True):
        result = []
        for i, t in enumerate(indicate):
            pos = t - len(self.SPECIAL_TOKENS)
            if t in self.SPECIAL_TOKENS:
                if t == self.SPECIAL_TOKEN_MASK:
                    if not source and self.n_only_input > 0:
                        data = torch.zeros(dataset.size(1)-self.n_only_input, dataset.size(2))
                    else:
                        data = torch.zeros(dataset.size(1), dataset.size(2))
                else:
                    data = torch.zeros(1, dataset.size(2))
            else:
                if not source and self.n_only_input > 0:
                    data = dataset[pos, :-self.n_only_input, :]
                else:
                    data = dataset[pos, :, :]

            data = data + self.pe[i].unsqueeze(dim=0).repeat_interleave(data.size(0), dim=0)
            result.append(data)
                
        result = torch.cat(result, dim=0)
        return result


    def get_source(self, times: torch.Tensor):
        src = self.get_data(times, self.source_dataset)
        return src
    
    def get_target(self, times: torch.Tensor):
        tgt = self.get_data(times, self.label_dataset, source=False)
        return tgt


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



class DenoisingDataset(Dataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
    """

    def __init__(
        self,
        vocab: HaeaVocab,
        seed,
        mask,
        mask_random,
        insert,
        rotate,
        permute_sentences,
        replace_length,
        mask_length,
        poisson_lambda,
        item_transform_func=None,
    ):

        self.vocab = vocab
        self.seed = seed
        self.mask_idx = vocab.SPECIAL_TOKEN_MASK
        self.mask_whole_word = None
        self.mask_ratio = mask
        self.random_ratio = mask_random
        self.insert_ratio = insert
        self.rotate_ratio = rotate
        self.permute_sentence_ratio = permute_sentences
        self.eos = vocab.SPECIAL_TOKEN_EOS
        self.item_transform_func = item_transform_func

        self.full_stop_index = self.eos

        self.dataset = self.make_dataset(len(vocab), vocab.max_len)

        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={mask_length}")
        if mask_length == "subword" and replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if mask_length == "span-poisson":
            _lambda = poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0
    

    def make_dataset(self, size, length):
        dataset = []
        
        for i in range(len(self.vocab.SPECIAL_TOKENS), size-length+2):
            start = i
            end = i + length
            data = [self.vocab.SPECIAL_TOKEN_BOS] + [x for x in range(start, end)] + [self.vocab.SPECIAL_TOKEN_EOS]
            dataset.append(data)

        return torch.tensor(dataset)


    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens.clone(), tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.mask_ratio > 0:
                source = self.add_word_mask(source, self.mask_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.SPECIAL_TOKEN_BOS
        assert source[-1] == self.vocab.SPECIAL_TOKEN_EOS

        source_dataset = self.vocab.get_source(source)
        target_dataset = self.vocab.get_target(target)

        return {
            "source": source,
            "source_dataset": source_dataset,
            "target": target,
            "target_dataset": target_dataset
        }

    def __len__(self):
        return len(self.dataset)

    def permute_sentences(self, source, p=1.0):
        full_stops = source == self.full_stop_index
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source
    

    def add_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return source

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()

        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]
        return source
    

    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result
