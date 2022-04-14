# Misspelling Semantics in Thai

This repo presents a fine-grained annotated corpus of misspelling in Thai, together with an analysis of misspelling intention and its possible semantics to
get a better understanding of the misspelling patterns observed in the corpus. 

In addition, we introduce two approaches to incorporate the semantics of misspelling: 
1) Misspelling Average Embedding (MAE) 
2) Misspelling Semantic Tokens (MST) 

Experiments on a sentiment analysis task confirm our overall hypothesis: additional semantics from misspelling can boost the micro F1 score up to 0.4-2%, while blindly normalising misspelling is harmful and suboptimal.

It is part of my paper "Misspelling Semantics in Thai" published at LREC2022, Marseille, France.

## Dataset
Our misspelling dataset is an extension of Wisesight Sentiment corpus [link](https://github.com/PyThaiNLP/wisesight-sentiment). Please see the original source for license and corpus detail.

Our new corpus is based on a sample of 3000 sentences from Wisesight training data. It is manually annotated by five recruited annotators. We employed
one-iterative annotation strategy where the annotators were asked to label misspellings according to our guideline. We then evaluated 100 samples and feedback to the annotators before asking them to re-label the data again. Each sentence was guaranteed to be annotated by three annotators. Each misspelling was labelled as intentional or unintentional. Further information about the guideline, please see our paper.

## Author
Myself, Pakawat Nakwijit, PhD Student @ QMUL

Supervisor, Matthew Purver

## License
This project is licensed under the terms of the MIT license except for the dataset which is under CC0 1.0 Universal.
