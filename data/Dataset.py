import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# import nltk
import sys


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y, vocab_npa, pad_token, unk_token):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        self.pad_token = pad_token
        self.unk_token = unk_token
        # For converting between words and idx
        self.word2idx = {word:idx for idx,word in enumerate(vocab_npa)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        # Data
        self.ids = []
        self.seq_lengths = []
        self.labels = []

        assert(len(x) == len(y))
        n_rows = len(x)  # may need df.shape[0] instead
        max_seq_length = 134
        # max_seq_length = max(df.iloc.map(len))

        # process each row
        for i in range(n_rows):
            # row = df.iloc[i]
            question_text = x[i]
            label = y[i]

            ids, seq_length = self.text_to_ids(text=question_text, pad_to_len=max_seq_length)
            self.ids.append(ids.reshape(-1))
            self.seq_lengths.append(seq_length)
            self.labels.append(label)

        # lemmatize
        # for question in df.question_text.tolist():
        #     nltk.

        # sanity checks
        assert len(self.ids) == n_rows
        assert len(self.seq_lengths) == n_rows
        assert len(self.labels) == n_rows


    def text_to_ids(self, text, pad_to_len):
        """returns a tensor of ids for each sentence, padded to the desired length"""
        words = text.strip().split()[:pad_to_len]  # just in case, truncate if more than pad_to_len

        # remove question mark from last word
        words[len(words) - 1] = words[len(words) - 1].replace('?', '')

        # add padding to reach proper length
        padding_to_add = pad_to_len - len(words)
        words.extend([self.pad_token] * padding_to_add)

        seq_length = len(words)  # must be after adding the padding
        
        # convert words to their ids, including padding
        for i in range(seq_length):
            if words[i] not in self.word2idx:
                words[i] = self.word2idx[self.unk_token]
            else:
                words[i] = self.word2idx[words[i]]
        
        return torch.Tensor(words).long(), seq_length  # not sure why nb converted to long


    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        Gets i-th sample of ids, seq_length, and label for this dataset item
        '''
        # reshape ids to collapse into 1D
        batch = {'input_ids': self.ids[i].reshape(-1),
            'seq_lengths': torch.tensor(self.seq_lengths[i]).long(),
            'labels': torch.tensor(self.labels[i], dtype=torch.float32)
        }
        return batch 


    # TODO: return the size of the dataset
    def __len__(self):
        return len(self.ids)
        # return self.sequences.shape[0]
