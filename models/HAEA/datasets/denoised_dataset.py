# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright Facebook, Inc. and its affiliates. Licensed under the MIT License as part of
# fairseq package.

import numpy as np
import torch
from torch.utils.data import Dataset
import math
import contextlib

class HaeaVocab:

    SPECIAL_TOKEN_BOS = 0
    SPECIAL_TOKEN_EOS = 1
    SPECIAL_TOKEN_MASK = 2
    SPECIAL_TOKEN_PAD = 3
    SPECIAL_TOKENS = [SPECIAL_TOKEN_BOS, SPECIAL_TOKEN_EOS, SPECIAL_TOKEN_MASK, SPECIAL_TOKEN_PAD]


    def __init__(self, source_dataset: torch.Tensor, label_dataset: torch.Tensor, n_only_input: int = 0):
        # dataset.shape = (time, var_len, hidden)
        self.source_dataset = source_dataset
        self.label_dataset = label_dataset
        self.n_only_input = n_only_input
        self.bos = torch.zeros(source_dataset.size(-1))
        self.eos = torch.zeros(source_dataset.size(-1))

    def __len__(self):
        return self.source_dataset.size(0) + len(self.SPECIAL_TOKENS)

    def get_data(self, indicate, dataset: torch.Tensor, source: bool = True):
        if not source:
            result = dataset[indicate, :-self.n_only_input, :]
        else:
            result = dataset[indicate, :, :]

        result = result.view(-1, result.size(-1))
        result = torch.cat([self.bos, result, self.eos])
        return result

    def get(self, times: torch.Tensor):
        times = times - len(self.SPECIAL_TOKENS)
        src = self.get_data(times, self.source_dataset)
        tgt = self.get_data(times, self.label_dataset, source=False)
        return src, tgt



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


class BARTDenoisingDataset(Dataset):
    def __init__(
        self,
        vocab: HaeaVocab,
        mask_whole_words,
        seed,
        args
    ):
        
        self.vocab = vocab
        self.seed = seed
        self.mask_idx = vocab.SPECIAL_TOKEN_MASK
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = args['mask_ratio']
        self.random_ratio = args['random_ratio']
        self.mask_length = args['mask_length']
        self.replace_length = args['replace_length']
        self.poisson_lambda = args['poisson_lambda']

        self.replace_length = self.replace_length
        if not self.replace_length in [-1, 0, 1]:
            raise (f'invalid arg: replace_length={self.replace_length}')
        if not self.mask_length in ['subword', 'word', 'span', 'span-poisson']:
            raise (f'invalid arg: mask-length={self.mask_length}')
        if self.mask_length == 'subword' and not self.replace_length in [0, 1]:
            raise (f'if using subwords, use replace-length=1 or 0')

        self.is_span_mask = (self.mask_length == 'span')
        self.mask_span_distribution = None
        if self.mask_length == 'span-poisson':
            _lambda = self.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= (k + 1)
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0
        torch.manual_seed(self.seed)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.vocab.SPECIAL_TOKEN_EOS
            source, target = tokens, tokens.clone()

            if self.mask_ratio > 0:
                if self.is_span_mask:
                    source = self.add_multiple_words_mask(source, self.mask_ratio)
                else:
                    source = self.add_whole_word_mask(source, self.mask_ratio)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.SPECIAL_TOKEN_BOS
        assert source[-1] == self.vocab.SPECIAL_TOKEN_EOS
        return {
            'id': index,
            'source': source,
            'target': target,
        }
    
    def make_dataset(self, size, length):
        dataset = torch.zeros(size, length)
        for i in range(size):
            start = i
            end = i + length - 1
            dataset[i] = torch.arange()


    def __len__(self):
        return len(self.dataset)

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0

        is_word_start[1] = 0  # exclude the first word. Label word
        # for i in range(1, self.tokens_to_keep+1):
        #     is_word_start[i] = 0

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
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
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
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

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
                    source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
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
                    source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_multiple_words_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        if num_to_mask == 0:
            return source

        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        start_index = word_starts.size(0)-num_to_mask
        if start_index < 1:
            print(source, is_word_start)
            return source

        mask_word_start_id = np.random.randint(start_index)

        source_length = source.size(0)
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc

        # keep first index, but replace it with [MASK], and delete remaining index
        source[word_starts[mask_word_start_id]] = self.mask_idx
        #assert mask_word_start_id+num_to_mask < word_starts.size(0)
        #assert (word_starts[mask_word_start_id].item()+num_to_mask) < source_length
        try:
            for ind in range(word_starts[mask_word_start_id]+1, word_starts[mask_word_start_id+num_to_mask]):
                to_keep[ind] = 0
        except IndexError:
            print("Index error", source, is_word_start)
            pass

        source = source[to_keep]
        return source