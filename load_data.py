import torch
from torchtext import data
import re
import dill


def token(s):
    tok = re.split(r'[^\w]', s)
    while True:
        try:
            tok.remove("")
        except:
            break
    return tok[:50]


def read_data(path_data, batch_size,p_vocab, save_vocab=None):
    max_strlen = 50
    tokenize = lambda x: token(x)
    TEXT = data.Field(sequential=True, tokenize=tokenize)
    LABEL = data.LabelField(dtype=torch.long, sequential=False)
    train = data.TabularDataset(path_data, format="csv", fields=[('src', TEXT),("trg",LABEL)], skip_header=True)
    tran_data, valid_data, = train.split()
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    """ 
    for i in train:
        if len(i.src) > max_strlen:
         max_strlen = len(i.src)

    for i in train:
        for j in range(len(i.src), max_strlen):
            i.src.append("<pad>")
    """

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((tran_data, valid_data, valid_data), batch_size=batch_size,
                                                            sort_key=lambda x: len(x.trg), repeat=False,
                                                            shuffle=False)
    print("num iter: ", len(train_iter))
    print("vocab size: ", len(TEXT.vocab))
    print("total data: ", len(train))
    
    if save_vocab == True:
        with open(p_vocab + "TEXT.Field", "wb")as f:
            dill.dump(TEXT, f)
    if save_vocab == True:
        with open(p_vocab + "LABEL.Field", "wb")as f:
            dill.dump(LABEL, f)
    return train_iter, valid_iter, test_iter, TEXT, LABEL, max_strlen

if __name__ == '__main__':
    path_data = "./data/data_vphc_train.csv"
    train_iter, valid_iter, test_iter, TEXT, LABEL, max_strlen = read_data(path_data, 10)
