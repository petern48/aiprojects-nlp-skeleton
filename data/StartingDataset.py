import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# import nltk


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path, vocab, max_seq_length, pad_token, unk_token):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''
        # Preprocess the data. These are just library function calls so it's here for you
        df = pd.read_csv(data_path)
        self.labels = self.df.target.tolist() # list of labels
        self.pad_token = pad_token
        self.unk_token = unk_token

        # For converting between words and idx
        self.word2idx = {word:idx for idx,word in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        # Data
        self.ids = []
        self.seq_lengths = []
        self.labels = []

        n_rows = len(df)  # may need df.shape[0] instead
        # process each row
        for i in range(n_rows):
            row = df.iloc[i]
            self.labels.append(row.label)
            ids, seq_length = self.text_to_ids(text=row.question_text, pad_to_len=max_seq_length)

        # for question in self.df.question_text.tolist():
        #     nltk.


    def text_to_ids(self, text, pad_to_len):
        """returns a tensor of ids, padded to the desired length"""
        words = text.strip().split()[:pad_to_len]  # just in case, truncate if more than pad_to_len
        seq_length = len(words)

        # add padding to reach proper length
        padding_to_add = pad_to_len - words
        words.extend([self.pad_token] * padding_to_add)

        # convert words to their ids
        for i in range(seq_length):
            if words[i] in self.word2idx:
                words[i] = self.word2idx(words[i])
            else:
                words[i] = self.word2idx(self.pad_token)

        return torch.Tensor(words).long(), seq_length  # not sure why nb converted to long


    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        Gets i-th sample of ids, seq_length, and label for this dataset item
        '''
        # reshape ids to collapse into 1D
        return self.ids[i].reshape(-1), \
            torch.tensor(self.seq_lengths[i]).long(), \
            torch.tensor(self.labels[i], dtype=torch.float32)  # might need .type(torch.FloatTensor) instead 


    # TODO: return the size of the dataset
    def __len__(self):
        return len(self.ids)
        # return self.sequences.shape[0]


def getEmbeddings(embeddings_path):
    """Reads from embeddings file path and returns the np array of vocab and pretrained embeddings."""

    vocab, embeddings = [], []

    with open(embeddings_path, 'r') as f:
        full_content = f.read() # read the file
        full_content = full_content.strip() # remove leading and trailing whitespace
        full_content = full_content.split('\n') # split the text into a list of lines
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0] # get the word at the start of the line
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]] # get the embedding of the word in an array
        # add the word and the embedding to our lists
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    # convert to numpy arrays
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')

    # make embeddings for these 2:
    # -> for the '<pad>' token, we set it to all zeros
    # -> for the '<unk>' token, we set it to the mean of all our other embeddings
    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)

    # insert embeddings for pad and unk tokens to embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

    # embedding layer should have dims = # of words in vocab  ×  # of dimensions in embedding
    # embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

    return vocab_npa, embs_npa
