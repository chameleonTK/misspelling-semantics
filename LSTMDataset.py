
from util import load_jsonl, norm_word, filter_by_mode
from torchtext.legacy import data
import torch
from tqdm import tqdm

LABELS = {
    "neg": 2,
    "neu": 1,
    "pos": 0,
    "q": 1 # used to be 3
}

def removeQuestion(label):    
  return LABELS[label]

def additionalToken(word):
  tokens = []
  w = norm_word(word)
  if "<lol>" in w:
    tokens.append("<lol>")
  elif "<rep>" in w:
    tokens.append("<rep>")
  elif w in MD:
      corr, mint = MD[w]
      if mint:
        tokens.append("<int>")
      else:
        tokens.append("<msp>")
  return tokens

def createMAEDataset(data, pre_segmented=False, mode=None, mst=False):
    output = []
    cnt, mstcnt = 0, 0

    segIdx = 0
    if mode=="corr":
        segIdx = 1  # ignore misspelling with MC

    for sent in data:
        newtokens = []
        misptokens = []
        if (mode=="misp" or mode=="corr") and len(sent["misp_tokens"])==0:
            continue

        if pre_segmented:
            for seg in sent["segments"]:
                for token in zip(seg[0], seg[1]):
                    newtokens.append(token[segIdx])
                    misptokens.append(token[1])
                    if mst:
                        msttokens = additionalToken(token[segIdx])
                        newtokens += msttokens
                        misptokens += msttokens

                    if token[0]!=token[1]:
                        mstcnt += 1
            
        else:
            for token in sent["tokenized"]:
                w = norm_word(token)
                detectedMsp = (w in MC)
                if detectedMsp:
                    corr, mint = MC[w]
                    misptokens.append(corr)
                    if mode=="corr":
                        token = corr
                        mstcnt += 1
                    newtokens.append(token)
                else:
                    misptokens.append(token)
                    newtokens.append(token)

                if detectedMsp and mst:
                    msttokens = additionalToken(token)
                    newtokens += msttokens
                    misptokens += msttokens


        cnt += len(newtokens)

        assert(len(newtokens)==len(misptokens))

        output.append({
            "category": sent["category"],
            "text": sent["text"],
            "tokenized": newtokens,
            "misp": misptokens
        })

    print(f"#Misp Tokens: {mstcnt} tokens; {(mstcnt)*100/cnt:.2f}%")
    return output


def addMSTTokens(data, pre_segmented=False, mode="misp"):
    output = []
    cnt, mstcnt = 0, 0
    for sent in data:
        newtokens = []

        if pre_segmented:
            segIdx = 0
            if mode=="corr":
                segIdx = 1
                
            for seg in sent["segments"]:
                for token in zip(seg[0], seg[1]):
                    newtokens.append(token[segIdx])
                    cnt += 1
                    
                    if token[0]==token[1]:
                        continue
                    
                    t = additionalToken(token[0])
                    newtokens += t
                    mstcnt += len(t)
        else:
            for token in sent["tokenized"]:
                newtokens.append(token)
                t = additionalToken(token)                    
                newtokens += t

                cnt += 1
                mstcnt += len(t)

        output.append({
            "category": sent["category"],
            "text": sent["text"],
            "tokenized": newtokens,
        })

    print(f"#New MST Tokens: {mstcnt} tokens; {(mstcnt)*100/cnt:.2f}%")
    return output

# ref: https://medium.com/@rohit_agrawal/using-fine-tuned-gensim-word2vec-embeddings-with-torchtext-and-pytorch-17eea2883cd
def set_wv_vectors(field, vectors, debug=False):
    W2V_SIZE = vectors.get_dimension()
    
    words = vectors.get_words()
    vocab_size = len(words)
    word2vec_vectors = []
    for token, idx in tqdm(field.vocab.stoi.items()):
        if idx==0:
            word2vec_vectors.append(torch.zeros(W2V_SIZE))
            continue
            
        word2vec_vectors.append(torch.FloatTensor(vectors[token]))

    field.vocab.set_vectors(field.vocab.stoi, word2vec_vectors, W2V_SIZE)

MC = load_jsonl(f".//test_mispelling_correction.jsonl")[0]
MD = load_jsonl(f"./train_mispelling_dection.jsonl")[0]

class LSTMDataset:
    def __init__(self, DIR):
        self.traindata = load_jsonl(f"{DIR}/WisesightSentiment/tokenized_train.jsonl")
        self.validdata = load_jsonl(f"{DIR}/WisesightSentiment/tokenized_valid.jsonl")
        self.testdata = load_jsonl(f"{DIR}/WisesightSentiment/tokenized_test-misp.jsonl")

       

        self.allTestdata = filter_by_mode(self.testdata)
        self.corrTestdata = filter_by_mode(self.testdata, "corr")
        self.mispTestdata = filter_by_mode(self.testdata, "misp")

        print("Datasets:", len(self.allTestdata), len(self.corrTestdata), len(self.mispTestdata))
        

        # TEXT = data.Field(sequential=True, lower=False)
        self.CATEGORY = data.Field(sequential=False, use_vocab=False, preprocessing=removeQuestion)
        self.TOKEN = data.Field(sequential=True, lower=False)

    def load(self):
        raw_datasets = {
            "train": self.traindata,
            "validation": self.validdata,
            "test": self.allTestdata,
            "test-corr": self.corrTestdata,
            "test-misp": self.mispTestdata
        }

        raw_fields = [
            # ('text', TEXT), 
            ('category', self.CATEGORY),
            ('tokenized', self.TOKEN)
        ]

        fields = {}
        for f in raw_fields:
            fields[f[0]] = f

        datasets = {}
        for k in raw_datasets:  
            examples = [data.Example.fromdict(d, fields=fields) for d in raw_datasets[k]]
            d = data.Dataset(examples, fields=raw_fields)
            datasets[k] = d
        
        return datasets
    

    def loadMAE(self):
        print("")
        print("Loading MAE datasets ...")
        mae_raw_datasets = {
            "test": createMAEDataset(self.testdata, pre_segmented=True),
            "test-corr": createMAEDataset(self.testdata, pre_segmented=True, mode="corr"),
            "test-misp": createMAEDataset(self.testdata, pre_segmented=True, mode="misp"),
        }

        mae_raw_fields = [
            ('category', self.CATEGORY),
            ('tokenized', self.TOKEN),
            ('misp', self.TOKEN),
        ]

        mae_fields = {}
        for f in mae_raw_fields:
            mae_fields[f[0]] = f

        print()
        MAEdatasets = {}
        for k in mae_raw_datasets:  
            print(f"Processed: {k}")
            examples = [data.Example.fromdict(d, fields=mae_fields) for d in mae_raw_datasets[k]]
            d = data.Dataset(examples, fields=mae_raw_fields)
            MAEdatasets[k] = d
        
        return MAEdatasets
    
    def loadMST(self):
        print("")
        print("Loading MST datasets ...")
        raw_datasets = {
            "train": self.traindata,
            "validation": self.validdata,
            "test": self.allTestdata,
            "test-corr": self.corrTestdata,
            "test-misp": self.mispTestdata
        }

        raw_fields = [
            # ('text', TEXT), 
            ('category', self.CATEGORY),
            ('tokenized', self.TOKEN)
        ]

        fields = {}
        for f in raw_fields:
            fields[f[0]] = f

        MSTdatasets = {}
        for k in raw_datasets:
            print(f"Processed: {k}")

            mode = "misp"
            if "corr" in k:
                mode = "corr"

            raw = addMSTTokens(raw_datasets[k], pre_segmented=k.startswith("test"), mode=mode)
            examples = [data.Example.fromdict(d, fields=fields) for d in raw]
            d = data.Dataset(examples, fields=raw_fields)
            
            MSTdatasets[k] = d
            print("")

        return MSTdatasets

    def loadBOTH(self):

        mae_raw_fields = [
            ('category', self.CATEGORY),
            ('tokenized', self.TOKEN),
            ('misp', self.TOKEN),
        ]

        mae_fields = {}
        for f in mae_raw_fields:
            mae_fields[f[0]] = f

        both_raw_datasets = {
            "test": createMAEDataset(self.testdata, pre_segmented=True, mst=True),
            "test-corr": createMAEDataset(self.testdata, pre_segmented=True, mode="corr", mst=True),
            "test-misp": createMAEDataset(self.testdata, pre_segmented=True, mode="misp", mst=True),
        }

        print()
        BOTHdataset = {}
        for k in both_raw_datasets:  
            print(f"Processed: {k}")
            examples = [data.Example.fromdict(d, fields=mae_fields) for d in both_raw_datasets[k]]
            d = data.Dataset(examples, fields=mae_raw_fields)
            # print(both_raw_datasets[k][0]["tokenized"])
            BOTHdataset[k] = d
        
        return BOTHdataset

    def buildVocab(self, datasets, MSTdatasets, wv):
        W2V_WINDOW = 5 
        W2V_MIN_COUNT = 0

        TOKEN = self.TOKEN
        CATEGORY = self.CATEGORY

        # TEXT.build_vocab(datasets["train"], min_freq=W2V_MIN_COUNT, )
        TOKEN.build_vocab(datasets["train"], datasets["validation"], datasets["test"], datasets["test-corr"], MSTdatasets["test"], min_freq=W2V_MIN_COUNT, )
        CATEGORY.build_vocab(datasets["train"])

        print("#Token",len(TOKEN.vocab))
        set_wv_vectors(TOKEN, wv)
    
    def iter(self, args, datasets, device):
        return data.BucketIterator.splits(
            datasets, 
            batch_size=args.batch_size, 
            
            # Sort all examples in data using `sort_key`.
            sort=True,
            sort_key=lambda x: len(x.tokenized),
            sort_within_batch=False,
            shuffle=True,
            
            device=device)



        
