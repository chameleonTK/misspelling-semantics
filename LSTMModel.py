from torchtext.legacy import data
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import time
import copy

def evaluate(loader, model, criterion, return_pred=False):
    model.eval()
    loader.sort = False
    loader.sort_within_batch = False
    loader.init_epoch()

    # calculate accuracy on validation set
    n_correct, n = 0, 0
    losses = []
    answers = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            answer = model(batch)
            answers.append((answer, batch.category))
            n_correct += (torch.max(answer, 1)[1].view(batch.category.size()) == batch.category).sum().item()
            n += answer.shape[0]
            loss = criterion(answer, batch.category)
            losses.append(loss.data.cpu().numpy())
    acc = 100. * n_correct / n
    loss = np.mean(losses)
    
    if not return_pred:
        return acc, loss
    
    
    predict = torch.cat([a for a,_ in answers])
    labels = torch.cat([a for _,a in answers])
    return acc, loss, predict, labels


class FeedForward(nn.Module):
    def __init__(self, d_model, d_out, d_ff=256, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 256
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_out)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Classifier(nn.Module):

    def __init__(self,
                 n_embed=10000,
                 d_embed=300,
                 d_hidden=256,
                 d_out=2,
                 dp=0.2,
                 embed_weight=None,
                 eow_idx=2):
        super(Classifier, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed = nn.Embedding(n_embed, d_embed)
        
        if embed_weight is not None:
            # embed_weight = inputs.vocab.vectors
            self.embed.weight.data.copy_(embed_weight)
            self.embed.weight.requires_grad = False
       
        # self.norm = Norm(d_embed)
        self.bilstm = torch.nn.LSTM(input_size=d_embed, hidden_size=d_hidden, num_layers=1, bidirectional=True, dropout=dp)
        self.ff = FeedForward(2*d_hidden, d_out, d_hidden)
        
        self.dropout =  nn.Dropout(dp)

    def forward(self, batch):
        tokens = batch.tokenized  
        # misp = batch.misp  
        label = batch.category

        w = self.embed(tokens)
        # m = self.embed(misp)
        # w = (w + m)/2
        o, (h, c) = self.bilstm(w)
        
        x = torch.cat((h[0,:,:], h[1,:,:]), dim=1)
        # x = self.norm(x)
        x = self.ff(self.dropout(x))
        
        return x


class LSTMModel:
    def __init__(self, TOKEN, CATEGORY, device):
        self.criterion = nn.CrossEntropyLoss()
        self.n_embed = len(TOKEN.vocab)
        self.d_out = len(CATEGORY.vocab)
        self.emb = TOKEN.vocab.vectors
        self.device = device
    
    def evaluate(self, iter, model, return_pred=False):
        return evaluate(iter, model, self.criterion, return_pred)

    def train_model(self, args, train_iter, validation_iter, test_iter):
        
        n_embed = self.n_embed
        d_out = self.d_out
        emb = self.emb
        device = self.device
        criterion = self.criterion


        model = Classifier(d_embed=args.d_embed, d_hidden=args.d_embed, d_out=d_out, dp=0.2, embed_weight=emb, n_embed=n_embed)
        model.to(device)
        best_model = model

        opt = optim.Adam(model.parameters(), lr=args.lr)

        acc, val_loss = evaluate(validation_iter, model, criterion)
        best_acc = acc

    #     print('epoch |   %        |  loss  |  avg   |val loss|   acc   |  best  | time | save |')
    #     print('val   |            |        |        | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(val_loss, acc, best_acc))

        iterations = 0
        last_val_iter = 0
        train_loss = 0
        start = time.time()
        
        train_stat = []
        with tqdm(total=args.epochs*len(train_iter)) as pbar:
            for epoch in range(args.epochs):
                train_iter.init_epoch()
                n_correct, n_total, train_loss = 0, 0, 0
                last_val_iter = 0
                # print(epoch, end=' ')

                for batch_idx, batch in enumerate(train_iter):
                    # switch model to training mode, clear gradient accumulators
                    model.train();
                    opt.zero_grad()

                    iterations += 1

                    # forward pass
                    answer = model(batch)
                    loss = criterion(answer, batch.category)

                    loss.backward();
                    opt.step()

                    train_loss += loss.item()
        #             print('\r {:4d} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
        #                 epoch, args.batch_size * (batch_idx + 1), len(train), loss.item(),
        #                         train_loss / (iterations - last_val_iter)), end='')
                    
                    stat = {
                        "epoch": epoch,
                        "step": iterations,
                        "train_loss": loss.item(),
                        "avg_loss": train_loss / (iterations - last_val_iter)
                    }

                    if iterations > 0 and iterations % args.dev_every == 0:
                        acc, val_loss = evaluate(validation_iter, model, criterion)
                        if acc > best_acc:
                            best_acc = acc
                            best_model = copy.deepcopy(model)
                            print("epoch", epoch)
                            # torch.save(model.state_dict(), args.save_path)


        #                 print(
        #                     ' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(
        #                         val_loss, acc, best_acc, (time.time() - start) / 60,
        #                         _save_ckp))

                        train_loss = 0
                        last_val_iter = iterations
                        stat["val_loss"] = val_loss
                        stat["acc"] = acc
                        stat["best_acc"] = best_acc
                        stat["time"] = (time.time() - start)
                
                    
                    train_stat.append(stat)
                    pbar.update(1)
        
        model = best_model
        acc, test_loss, predict, labels = evaluate(test_iter, model, criterion, return_pred=True)
        print(acc, test_loss)

        output = []
        # _predict = predict.cpu().numpy()
        # _labels = labels.cpu().numpy()
        # for idx, t in enumerate(test):
        #     output.append({
        #         "text": t.text,
        #         "label": t.category,
        #         # "tokens": json.dumps(t.tokens, ensure_ascii=False),
        #         "predict": json.dumps(_predict[idx].tolist(), ensure_ascii=False),
        # #         "_label": _labels[idx]
        #     })

        output = pd.DataFrame(output)
        # output.to_csv(f"Mispelling/Outputs/{expname}_{tokenType}.csv", index=False)
        
        train_stat = pd.DataFrame(train_stat)
        # train_stat.to_csv(f"Mispelling/Outputs/{expname}_{tokenType}_train_stat.csv", index=False)
        return model, output, train_stat
