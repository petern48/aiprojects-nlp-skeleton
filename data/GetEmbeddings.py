import numpy as np


def getEmbeddings(embeddings_path, pad_token, unk_token):
    """Reads from embeddings file path and returns the np array of vocab and pretrained embeddings."""

    vocab, embeddings = [], []

    with open(embeddings_path, 'r') as f:
        full_content = f.read() # read the file
        full_content = full_content.strip() # remove leading and trailing whitespace
        full_content = full_content.split('\n') # split the text into a list of lines

    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0] # get the word at the start of the line
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]  # get the embedding of the word in an array
        # add the word and the embedding to our lists
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    # convert to numpy arrays
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    # add pad and unk token
    vocab_npa = np.insert(vocab_npa, 0, pad_token)
    vocab_npa = np.insert(vocab_npa, 1, unk_token)

    # make embeddings for these 2:
    # -> for the '<pad>' token, we set it to all zeros
    # -> for the '<unk>' token, we set it to the mean of all our other embeddings
    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)

    # insert embeddings for pad and unk tokens to embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    # embedding layer should have dims = # of words in vocab  ×  # of dimensions in embedding
    # (rows, cols)
    # (400002, num rows)
    # layer x column vec input.  so input should be same dims as layer's num of cols

    return vocab_npa, embs_npa
