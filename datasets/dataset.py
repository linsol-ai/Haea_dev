from torch.utils.data import Dataset
import torch
import random
import numpy as np


class HAEADataset(Dataset):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3

    special_tokens = [0, 1, 2, 3, 4]


    def __init__(self, dataset: torch.Tensor, time_len: int):
        # dataset.shape = (time, var_len, hidden)
        self.dataset = dataset
        self.var_len = dataset.size(1)
        self.time_len = time_len


    def __len__(self):
        return self.dataset.size(0)


    def make_dataset(self):
        dataset = []
        for t in range(self.dataset.size(0)):
            src = [(t, v) for v in range(self.var_len)]
            tgt = []
            for t_


    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_variable_seq(t1)
        t2_random, t2_label = self.random_variable_seq(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [ [self.sos_index, t1[0][1], t1[0][2]] ] + t1_random + [ [self.eos_index, t1[0][1], t1[0][2]] ]
        t2 = t2_random + [ [self.eos_index, t2[0][1], t2[0][2]] ]

        t1_label = [ [self.pad_index, t1[0][1], t1[0][2]] ] + t1_label + [ [self.pad_index, t1[0][1], t1[0][2]] ]
        t2_label = t2_label + [ [self.pad_index, t2[0][1], t2[0][2]] ]


        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding1 = [ [self.pad_index, 0, 0] for _ in range(self.seq_len - len(bert_input))]
        padding2 = [ self.pad_index for _ in range(self.seq_len - len(bert_input))]

        predict = []

        for i, token in enumerate(bert_label):
            key = token[0]
            if key != self.pad_index:
                predict.append(i)
        
        predict.extend([0 for _ in range(self.seq_len - len(predict))])
        bert_input.extend(padding1), bert_label.extend(padding1), segment_label.extend(padding2)

        bert_input, _, val_seq = self.pos_to_pos(bert_input)
        bert_label, time_seq, _ = self.pos_to_pos(bert_label)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "predict": predict,
                  "time_seq": time_seq,
                  "val_seq": val_seq,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}
    

    def pos_to_pos(self, input):
        output = []
        time_seq = []
        val_seq = []

        for i, token in enumerate(input):
            key = token[0]
            week = token[2]

            time_seq.append(week)
            val_seq.append(key)
            output.append((key, token[1]))

        return output, time_seq, val_seq


    def random_variable_seq(self, var_seq):
        output_label = []
        tokens = var_seq #(변수 개수, 2)

        for i, var_token in enumerate(var_seq):
            key = var_token[0]
            time = var_token[1]

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i][0] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    r = random.randrange(len(self.dataset))
                    tokens[i][1] = self.dataset[r][0]
                    
                output_label.append([key, time, var_token[2]])
            else:
                output_label.append([self.pad_index, time, var_token[2]])

        return tokens, output_label


    def random_sent(self, index):
        time_idx = self.dataset[index][0]
        t1, t2 = self.get_corpus_lines(index)
        if index < len(self.dataset)-1 and time_idx < self.dataset_size-1 and self.dataset[index+1][0]-time_idx == 1:
            # output_text, label(isNotNext:0, isNext:1)
            if random.random() > 0.5:
                return t1, t2, 1
            else:
                line = self.get_random_line()
                if line[0][1] == index + 1:
                    return t1, line, 1
                return t1, line, 0
        else:
            line = self.get_random_line()
            return t1, line, 0


    def get_corpus_lines(self, index):
        current = self.dataset[index]
        if index < len(self.dataset)-1:
            next = self.dataset[index+1]
            # (변수, 시간, 주차)
            return [ [v + len(self.special_tokens), current[0], current[1]] for v in self.var_list], [ [v + len(self.special_tokens), next[0], next[1]] for v in self.var_list]
        else:
            return [ [v + len(self.special_tokens), current[0], current[1]] for v in self.var_list], []

    def get_random_line(self):
        r = random.randrange(len(self.dataset))
        return [ [v + len(self.special_tokens), self.dataset[r][0], self.dataset[r][1]] for v in self.var_list]



