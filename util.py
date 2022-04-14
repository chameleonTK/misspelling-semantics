import json
from itertools import groupby
import itertools

def load_jsonl(fname):
    fin = open(fname, encoding="utf-8")
    data = []
    for line in fin:
        d = json.loads(line.strip())
        data.append(d)

    return data

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as fo:
        for idx, d in enumerate(data):
            fo.write(json.dumps(d, ensure_ascii=False))
            fo.write("\n")


def norm_word(word, haha=True, notoken=False):
    groups = [list(s) for _, s in groupby(word)]
    ch = []
    extraToken = ""
    for g in groups:
        if len(g)>=3:
            if g[0]=="5" and haha:
              extraToken = "<lol>"
              ch.append('555')  
            else:
              extraToken = "<rep>"
              ch.append(g[0])  
        else:
            ch += g
    word = "".join(ch)+extraToken
    if notoken:
      word = "".join(ch)

    return word

def filter_by_mode(data, mode=None):
  output = []
  for sent in data:
    if mode is None:
      tokenized = [seg[0] for seg in sent["segments"]]
    elif mode=="corr":
      tokenized = [seg[1] for seg in sent["segments"]]
      if len(sent["misp_tokens"])==0:
        continue
    else:
      tokenized = [seg[0] for seg in sent["segments"]]
      if len(sent["misp_tokens"])==0:
        continue
    
    tokenized = list(itertools.chain(*tokenized))
  
    output.append({
        "category": sent["category"],
        "text": sent["text"],
        "tokenized": tokenized,
        "segments": sent["segments"]
    })

  return output
    