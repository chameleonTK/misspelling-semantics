import json
import pandas as pd
from util import load_jsonl, norm_word, filter_by_mode
from SequenceClassificationDataset import SequenceClassificationDataset
from tqdm import tqdm
import itertools
from datasets import Dataset

from itertools import groupby
from collections import defaultdict

LABELS = {
    "neg": 2,
    "neu": 1,
    "pos": 0,
    "q": 1 # used to be 3
}

MC = load_jsonl(f".//test_mispelling_correction.jsonl")[0]
MD = load_jsonl(f"./train_mispelling_dection.jsonl")[0]

class CustomLabelEncoder():
    def __init__(self):
        pass

    def transform(self, labels):
        return [LABELS[l] for l in labels]

def get_dict_val(root, keys):
    if type(keys) == str:
        return root[keys]
    elif type(keys) == list:
        _results = []
        for item in root[keys[0]]:
            _results.append(item[keys[1]])
        return _results
    
    return None

def preprocessing(d, tokenizer, preSegmented=False, mode=None, mst=False):
    max_length = 400
    custom_label_encoder = CustomLabelEncoder()
    labels = get_dict_val(d, "category")

    labels = custom_label_encoder.transform(labels)

    input_ids = []
    misp_ids = []
    attention_masks = []
    unk = tokenizer.convert_tokens_to_ids(["<unk>"])[0]

    cnt = defaultdict(int)
    sents = []
    if not preSegmented:
        texts = get_dict_val(d, "tokenized")
        for tokens in tqdm(texts):
            tokens = [(t, t) for t in tokens]
            sents.append(tokens)
        
    else:
        texts = get_dict_val(d, "segments")
        for segments in tqdm(texts):
            s = [list(zip(seg[0], seg[1])) for seg in segments]
            tokens = list(itertools.chain(*s))
            sents.append(tokens)

    for tokens in sents:
        # if mode is None, ignore corr
        cnt["tokens"] += len(tokens)
        
        reftokens = [t[0] for t in tokens]
        if mst:
            tokens = [
                (norm_word(t[0], notoken=True).lower(), None if t[1] is None else norm_word(t[1], notoken=True).lower())
                for t in tokens
            ]
        else:
            tokens = [
                (t[0].lower(), None if t[1] is None else t[1].lower())
                for t in tokens
            ]

        misptokens = [t[0] for t in tokens]
        corrtokens = [t[0] for t in tokens]
        
        if mode=="corr":
            misptokens = [t[1] for t in tokens]
            corrtokens = [t[1] for t in tokens]
            reftokens = [t[1] for t in tokens]

        elif mode=="mae":
            misptokens = [t[0] for t in tokens]
            corrtokens = [t[1] for t in tokens]
      
      
        midx = tokenizer.convert_tokens_to_ids(misptokens)
        cidx = tokenizer.convert_tokens_to_ids(corrtokens)
        assert(len(midx)==len(cidx))

        newmisptokens = []
        newcorrtokens = []
        for i in range(len(midx)):
            if midx[i]==unk:
                t = tokenizer.tokenize("_"+misptokens[i])[1:]
                if misptokens[i]==corrtokens[i]:
                    newmisptokens += t
                    newcorrtokens += t
                else:
                    newmisptokens += t
                    tx = tokenizer.tokenize("_"+corrtokens[i])[1:]

                    if len(tx) > 0:
                        newcorrtokens += [tx[0] for j in range(len(t))]
                    else:
                        newcorrtokens += t
            else:
                newmisptokens.append(misptokens[i])
                newcorrtokens.append(corrtokens[i])
            
            if misptokens[i]!=corrtokens[i]:
                cnt["misspelling"] += 1

            if mst:
                norm = norm_word(reftokens[i])
                if "<rep>" in norm:
                    newmisptokens.append("<rep>")
                    newcorrtokens.append("<rep>")
                    cnt["<rep>"] += 1
                elif "<lol>" in norm:
                    newmisptokens.append("<lol>")
                    newcorrtokens.append("<lol>")
                    cnt["<lol>"] += 1
                
                if norm in MD:
                    corr, mint = MD[norm]
                    if mint:
                        newmisptokens.append("<int>")
                        newcorrtokens.append("<int>")
                        cnt["<int>"] += 1
                    else:
                        newmisptokens.append("<msp>")
                        newcorrtokens.append("<msp>")
                        cnt["<msp>"] += 1
            
        assert(len(newmisptokens)==len(newcorrtokens))
        cnt["extra"] += (len(newmisptokens) - len(misptokens))
            
        # words = newwords
        newmisptokens = ['<s>'] + newmisptokens[0:max_length-2] + ['</s>']
        newcorrtokens = ['<s>'] + newcorrtokens[0:max_length-2] + ['</s>']
        
        midx = tokenizer.convert_tokens_to_ids(newmisptokens)
        cidx = tokenizer.convert_tokens_to_ids(newcorrtokens)

        mask = [1 for i in midx]
            
        input_ids.append(midx)
        misp_ids.append(cidx)
        attention_masks.append(mask)

        #if len(input_ids) > 50:
        #    break
    
    #labels = labels[0:50]
    for k in cnt:
        print("Number of", k, cnt[k])

    return SequenceClassificationDataset(
        tokenizer=tokenizer,
        data_dir=None,
        max_length=max_length,
        input_ids=input_ids,
        misp_ids=misp_ids,
        attention_masks=attention_masks,
        labels=labels
    )

class WangchanBERTaDataset:
    def __init__(self, DIR, tokenizer, fewshot=False):
        if fewshot:
            self.traindata = load_jsonl(f"{DIR}/tokenized_train-misp-3000.jsonl")
        else:
            self.traindata = load_jsonl(f"{DIR}/tokenized_train.jsonl")

        self.fewshot = fewshot
        self.validdata = load_jsonl(f"{DIR}/tokenized_valid.jsonl")
        self.testdata = load_jsonl(f"{DIR}/tokenized_test-misp.jsonl")

        self.allTestdata = filter_by_mode(self.testdata)
        self.corrTestdata = filter_by_mode(self.testdata, "corr")
        self.mispTestdata = filter_by_mode(self.testdata, "misp")

        print("Datasets:", len(self.allTestdata), len(self.corrTestdata), len(self.mispTestdata))
        self.tokenizer = tokenizer

    def preprocess(self, mst=False):
        raw_datasets = {
            "train": self.traindata,
            "validation": self.validdata,
            "test": self.allTestdata,
            "test-corr": self.corrTestdata,
            "test-misp": self.mispTestdata,
            "test-mae": self.mispTestdata,
            "test-all-mae": self.allTestdata,
        }

        dataset_split = {}
        for split_name in raw_datasets:
            print("Tokenizing:", split_name)
            d = pd.DataFrame(raw_datasets[split_name])
            d = Dataset.from_pandas(d)
            preSegmented = split_name.startswith("test")
            if self.fewshot:
                preSegmented = (split_name.startswith("test") or split_name.startswith("train"))
                
            mode = None
            if "corr" in split_name:
                mode = "corr"
            elif "misp" in split_name:
                mode = "misp"
            elif "mae" in split_name:
                mode = "mae"

            cc = 0
            dataset_split[split_name] = preprocessing(d, self.tokenizer, preSegmented=preSegmented, mode=mode, mst=mst)
        #     for d in dataset_split[split_name]:
        #         print(raw_datasets[split_name][cc]["text"])
        #         print(raw_datasets[split_name][cc]["tokenized"])
        #         print(self.tokenizer.convert_ids_to_tokens(d["input_ids"]))
        #         print(self.tokenizer.convert_ids_to_tokens(d["misp_ids"]))
        #         print("")
        #         if cc > 5:
        #             break
        #         cc += 1
        
        # assert(False)
        
        return dataset_split
