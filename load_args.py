import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='path to a pretrained model')
    parser.add_argument('--embs_path', type=str, default='', help='path to embeddings file')
    parser.add_argument('--data_file', type=str, default='', help='path to data file for training')
    parser.add_argument('--using_notebook', action="store_true", default=False, help='set if using notebook to use proper tqdm')

    args = parser.parse_args()
    return args
