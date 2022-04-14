import torch.nn as nn
import torch

class MAEClassifier(nn.Module):

    def __init__(self, refModel):
        super(MAEClassifier, self).__init__()

        self.ref = refModel

    def forward(self, batch):
        tokens = batch.tokenized  
        misp = batch.misp  
        label = batch.category

        w = self.ref.embed(tokens)
        m = self.ref.embed(misp)
        w = (w + m)/2
        o, (h, c) = self.ref.bilstm(w)
        
        x = torch.cat((h[0,:,:], h[1,:,:]), dim=1)
        x = self.ref.ff(self.ref.dropout(x))
        return x