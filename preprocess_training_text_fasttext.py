
import fasttext

import pandas as pd
from util import load_jsonl, norm_word


from bs4 import BeautifulSoup
from tqdm import tqdm


def ignore(m, c):
  if c.endswith(" ๆ"):
    return True

  if m.replace(" ", "")==c.replace(" ", ""):
    return True

  if ("ฯ" in c) and (m not in ["ๆลๆ", "พณฯท่าน", "ฯล"]):
    return True

  if ("." in m) or ("." in c):
    return True
  
  return False

def process_dom(dom):
  tokens = []
  cntMisp, cntIgnored = 0, 0
  if dom.name is None:
    words = str(dom).split("|")
    for w in words:
      w = w.strip()
      if len(w)==0:
        continue
      tokens.append((w, w))
  elif dom.name=="ne":
    words = dom.text.strip().split("|")
    for w in words:
      w = w.strip()
      if len(w)==0:
        continue
      tokens.append((w, w))

  elif dom.name=="msp":
    m = dom.text.replace("|", "").strip()
    c = dom["value"].strip()
    # assert("|" not in m)

    cntMisp += 1
    if ignore(m, c):
      tokens.append((m, m))
      cntIgnored += 1
    else:
      tokens.append((m, c))
      mispTokens.add((m, c))
  elif dom.name=="compound":
    for child in dom.children:
      tkn, cms, cig = process_dom(child)
      tokens += tkn
      cntMisp += cms
      cntIgnored += cig
  else:
    print(dom)
    raise(f"Unknown Tag: {dom.name}")

  return tokens, cntMisp, cntIgnored


if __name__ == '__main__':

    DIR = "Datasets/"
    MD = load_jsonl(f"{DIR}/../train_mispelling_dection.jsonl")[0]

    # VISTEC-TP-TH-2021
    print("Processing VISTEC-TP-TH-2021")
    sent = []
    mispTokens = set()
    cntIgnored = 0
    cntMisp = 0
    cntToken = 0
    with open(f"{DIR}/VISTEC-TP-TH-2021/train/VISTEC-TP-TH-2021_train_proprocessed.txt", encoding="utf-8") as fin:
        for line in tqdm(fin, total=40000):
            line = line.strip()
            s = BeautifulSoup("<div id='text'>"+line+"</div>")
            tokens = []
            for dom in s.find("div", {"id": "text"}).children:
                tkn, cms, cig = process_dom(dom)
                tokens += tkn
                cntMisp += cms
                cntIgnored += cig

                cntToken += len(tkn)
                    
            sent.append(tokens)
            # misp.append(" ".join([t[0] for t in tokens]))
            # corr.append(" ".join([t[1] for t in tokens]))

    print()
    print("#Tokens:", cntToken)
    print("#Misspelling Tokens:", cntMisp)
    print("#Misspelling Tokens[Skip]:", cntIgnored, cntIgnored*100/cntMisp)


    with open(f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_misp.txt", "w", encoding="utf-8") as fout:
        for tokens in sent:
            s = " ".join([t[0] for t in tokens])
            fout.write(s+"\n")

    with open(f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_corr.txt", "w", encoding="utf-8") as fout:
        for tokens in sent:
            s = " ".join([t[1] for t in tokens])
            fout.write(s+"\n")

    with open(f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_MST.txt", "w", encoding="utf-8") as fout:
        cnt = 0
        for tokens in sent:
            s = []
            for t in tokens:
                if t[1]!=t[0]:
                    s.append(t[0])
                    
                    w = norm_word(t[0])
                    if "<lol>" in w:
                        s.append("<lol>")
                    elif "<rep>" in w:
                        s.append("<rep>")
                    else:
                        if w in MD:
                            corr, mint = MD[w]
                            if mint:
                                s.append("<int>")
                            else:
                                s.append("<msp>")
                        else:
                            s.append("<msp>")
                else:
                    s.append(t[0])
            # break
            s = " ".join(s)
            
            fout.write(s+"\n")


    print()
    print()
    
    # Wisesight Train
    print("Processing Wisesight Train")

    wisesight = load_jsonl(f"{DIR}/WisesightSentiment/tokenized_train.jsonl")
    with open(f"{DIR}/wisesight_train_fasttext_training_misp.txt", "w", encoding="utf-8") as fout:
        for sent in wisesight:
            s = " ".join(sent["tokenized"])
            fout.write(s+"\n")

    cntToken = 0
    cntMisp = 0
    with open(f"{DIR}/wisesight_train_fasttext_training_MST.txt", "w", encoding="utf-8") as fout:
        for sent in wisesight:
            s = []
            for t in sent["tokenized"]:
                s.append(t)
                w = norm_word(t)
                cntToken += 1
                
                if "<lol>" in w:
                    s.append("<lol>")
                    cntMisp += 1 
                elif "<rep>" in w:
                    s.append("<rep>")
                    cntMisp += 1
                else:
                    if w in MD:
                        corr, mint = MD[w]
                        if mint:
                            s.append("<int>")
                        else:
                            s.append("<msp>")
                        
                        cntMisp += 1

            s = " ".join(s)
            fout.write(s+"\n")

    print("#Tokens:", cntToken)
    print("#Misspelling Tokens:", cntMisp, cntMisp*100/cntToken)


