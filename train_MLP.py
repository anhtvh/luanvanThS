from load_data import read_data
from model.model_MLP import MLPClassifier
from model.model_MLP_noLM import MLP
import argparse
import torch
import torchwordemb
from  sklearn.metrics import f1_score
import torch.nn.functional as F
from  torch import nn
from torch.autograd import Variable
import time

def embed(x, vocab, vec,TEXT):

    if torch.cuda.is_available():
        x_new = torch.cuda.FloatTensor(x.shape[0], x.shape[1], 300)
    else:
        x_new = torch.FloatTensor(x.shape[0], x.shape[1], 300)
    for i in range(x.shape[0]):
        if torch.cuda.is_available():
            sentence = torch.cuda.FloatTensor(x.shape[1],300)
        else:
            sentence = torch.FloatTensor(x.shape[1], 300)
        for j in range(x.shape[1]):
            word = TEXT.vocab.itos[x[i][j].item()]
            if word == "<pad>":
                emb = torch.zeros(300)
            else:
                try:
                    emb = vec[vocab[word]]
                except:
                    emb = torch.randn(300)
            sentence[j]= emb
        if torch.cuda.is_available():
            x_new[i] = torch.cuda.FloatTensor(sentence)
        else:
            x_new[i] = torch.FloatTensor(sentence)
    return x_new

def read_embed(embed_path,LM):

    if LM == "glove":
        vocab, vec = torchwordemb.load_glove_text(embed_path)
    else:
        vocab, vec = torchwordemb.load_word2vec_text(embed_path)

    return vocab, vec





def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(model, train_iter, loss_fn, epoch, lr, p_log,optim, use_LM = None, vocab=None, vec=None, TEXT=None ):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_f1 = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.src.transpose(0, 1)
        target = batch.trg

        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        if use_LM == True:
            input_sen = embed(text, vocab, vec, TEXT)
        else:
            input_sen = text

        probs = model(input_sen)
        optim.zero_grad()
        loss = loss_fn(probs, target)
        loss.backward()
        clip_gradient(model, lr)
        optim.step()
        num_corrects = (torch.max(probs, 1)[1].view(target.size()).data == target.data).sum()
        acc = 100.0 * num_corrects / len(batch)
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        pre = torch.max(probs, 1)[1].view(target.size()).data
        f1 = f1_score(pre.data.to("cpu"),target.data.to("cpu"),average="weighted")
        total_f1 += f1
        if idx % 1 == 0:
            with open(p_log, "a") as f:
                f.write("Epoch: " + str(epoch + 1) + ", Idx: " + str(idx + 1) + ", Training Loss: " + str(
                    loss.item()) + ", Training Accuracy: " + str(acc.item()) + " F1_score: " + str(f1))
                f.write("\n")
            print(
                f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%, F1_core: {f1: .2f}% ')
        steps += 1
    return total_epoch_acc / steps, total_epoch_loss / steps, total_f1/steps


def eval(model, valid_iter, loss_fn, use_LM = None, vocab=None, vec=None, TEXT=None ):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_f1 = 0
    model.eval()

    for idx, batch in enumerate(valid_iter):
        text = batch.src.transpose(0, 1)
        target = batch.trg

        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        if use_LM == True:
            input = embed(text, vocab, vec, TEXT)

        else:
            input = text

        probs = model(input)
        loss = loss_fn(probs, target)
        num_corrects = (torch.max(probs, 1)[1].view(target.size()).data == target.data).sum()

        acc = 100.0 * num_corrects / len(batch)
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        pre = torch.max(probs, 1)[1].view(target.size()).data
        f1 = f1_score(pre.data.to("cpu"), target.data.to("cpu"), average="weighted")
        total_f1 += f1

    return total_epoch_acc / len(valid_iter), total_epoch_loss / len(valid_iter), total_f1 / len(valid_iter)

def checkmodel(LM):
    if LM == "fasttext":
        return "./LM/fastext/wiki/vi_wiki_fasttext_300.txt", "./LM/fastext/wiki/MLP/log.log","./LM/fastext/wiki/MLP/ckpt.pth","./LM/fastext/wiki/MLP/"
    elif LM == "glove":
        return "./LM/glove/wiki/vetors.txt", "./LM/glove/wiki/MLP/log.log", "./LM/glove/wiki/MLP/ckpt.pth", "./LM/glove/wiki/MLP/"
    elif LM == "w2v":
        return "./LM/w2v/wiki/vi_wiki_word2vec_300.txt", "./LM/w2v/wiki/MLP/log.log", "./LM/w2v/wiki/MLP/ckpt.pth", "./LM/w2v/wiki/MLP/"
    else:
        return "", "./LM/non/wiki/MLP/log.log", "./LM/non/wiki/MLP/ckpt.pth", "./LM/non/wiki/MLP/"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP text classificer')
    parser.add_argument('-LM', type=str, default="non", help='enter LM: w2v,fasttext,glove')
    parser.add_argument('-batch', type=int, default=80, help='enter bach_size')
    parser.add_argument('-data', type=str, default="data_vphc_train.csv", help='enter name file datatrain')
    args = parser.parse_args()
    LM = str(args.LM)
    p_LM, p_log, p_model, p_vocab = checkmodel(LM)

    batch_size = int(args.batch)

    path_data = "./data/" + args.data
    train_iter, valid_iter, test_iter, TEXT, LABEL, max_strlen = read_data(path_data, batch_size,p_vocab,save_vocab=True)
    if LM != "non":
        vocab, vec = read_embed(p_LM, LM)

    vocab_size = len(TEXT.vocab)
    embedding_length = 300
    output_size = len(LABEL.vocab)
    keep_probab = 0.5
    lr = 0.001
    start_time = time.time()
    if LM != "non":
        model = MLPClassifier(output_size, keep_probab, max_strlen)
    else:
        model = MLP(output_size, keep_probab, max_strlen, embedding_length, vocab_size)

    if torch.cuda.is_available():
        model.cuda()

    loss_fn = F.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    p_log_average = p_log.split(".log")[0] + "_average.log"
    p_save = p_model

    p_log_hyper = p_log.split(".log")[0] + "_hyper.log"
    p_save = p_model
    with open(p_log_hyper, "a") as f:
        f.write("vocab_size ")
        f.write(vocab_size.__str__())
        f.write("\n embedding_length")
        f.write(embedding_length.__str__())
        f.write("\n keep_probab")
        f.write(keep_probab.__str__())
        f.write("\n output_size")
        f.write(output_size.__str__())
        f.write("\n lr")
        f.write(lr.__str__())
        
    
    for epoch in range(1000):
        if LM != "non":
            train_acc, train_loss, train_f1 = train(model, train_iter, loss_fn,  epoch, lr, p_log, optim, use_LM=True, vocab=vocab, vec=vec, TEXT=TEXT)
            val_acc, val_loss, val_f1 = eval(model, valid_iter, loss_fn, use_LM=True, vocab=vocab, vec=vec, TEXT=TEXT)
        else:
            train_acc, train_loss, train_f1 = train(model, train_iter, loss_fn, epoch, lr, p_log, optim)
            val_acc, val_loss, val_f1 = eval(model, valid_iter, loss_fn)
        with open(p_log_average, "a") as f:
            f.write("Epoch " + str(epoch) + " Train loss " + str(train_loss) + " Train Acc: " + str(
                train_acc) + " train_F1 " + str(train_f1) + " Val. Loss " + str(val_loss) + " Val. Acc " + str(val_acc) + " Val_F1 " + str(val_f1) + " Time: " + str(int((time.time()-start_time)/60)))
            f.write("\n")
        print(
            f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Train_F1 {train_f1:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val_F1 {val_f1:.2f}%')
        torch.save(model.state_dict(), p_save)
        if val_loss < 0.9 :
            p_save = str(p_model.split(".pth")[0] + "_epoch_" + str(epoch + 1) + ".pth")

    if LM != "non":
        test_acc, test_loss, test_f1 = eval(model, valid_iter, loss_fn, use_LM=True, vocab=vocab, vec=vec, TEXT=TEXT)
    else:
        test_acc, test_loss, test_f1 = eval(model, valid_iter, loss_fn)
    with open(p_log_average, "a") as f:
        f.write("\n")
        f.write("Test Loss: " + str(test_loss) + " Test Acc: " + str(test_acc) + " Test_F1: " + str(test_f1))
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
