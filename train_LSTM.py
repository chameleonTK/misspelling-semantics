import numpy as np
import random
import torch
import os

import torch
import sys
from argparse import ArgumentParser


from LSTMDataset import LSTMDataset
from LSTMModel import LSTMModel
from MAEClassifier import MAEClassifier
import fasttext

import gc

def set_random_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)    
    np.random.seed(seed)
    np.random.RandomState(seed)

    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #seed all gpus    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser(description='LSTM')
#     parser.add_argument('mode', type=str, help = 'tokenizing mode ')
    parser.add_argument('--epochs', type=int, default=50, help = 'epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=100)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--save_path', type=str, default='results', help='path to save the model')
    
    try:
        args = parser.parse_args([])
    except:
        parser.print_help()
        sys.exit(1)

    return args

if __name__ == '__main__':
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    DIR = "Datasets"
    if len(sys.argv) < 2:
        raise Exception("Please specify wv")

    wvDIR = sys.argv[1]
    wv = fasttext.load_model(wvDIR)

    # pretrained fasttext wv is fixed, add randomness 
    # other wv(s) are not fixed, don't add randomness 
    if "cc.th.300.bin" not in wvDIR:
        set_random_seed(0)

    args = get_args()
    args.epochs = 100
    args.batch_size = 256
    args.dev_every = 50
    args.d_embed = wv.get_dimension()

    DATASETS = LSTMDataset(DIR)
    datasets = DATASETS.load()
    MAEdatasets = DATASETS.loadMAE()
    MSTdatasets = DATASETS.loadMST()
    BOTHdataset = DATASETS.loadBOTH()

    DATASETS.buildVocab(datasets, MSTdatasets, wv)

    # Reclaim RAM
    del wv
    gc.collect()

    train_iter, validation_iter, test_iter = DATASETS.iter(args, (datasets["train"], datasets["validation"], datasets["test"]), device)

    # Train Model

    MODEL = LSTMModel(DATASETS.TOKEN, DATASETS.CATEGORY, device)

    print("Arguments:", args)

    print("")
    print("Training ...")
    model, output, train_stat = MODEL.train_model(args, train_iter, validation_iter, test_iter)

    print("")
    print("Testing ...")
    test_iter, corr_iter, misp_iter = DATASETS.iter(args, (datasets["test"], datasets["test-corr"], datasets["test-misp"]), device)

    acc, test_loss, predict, labels = MODEL.evaluate(test_iter, model, return_pred=True)
    print("Test Accuracy: ", acc)


    acc, test_loss, predict, labels = MODEL.evaluate(corr_iter, model, return_pred=True)
    print("Corr Accuracy: ", acc)

    acc, test_loss, predict, labels = MODEL.evaluate(misp_iter, model, return_pred=True)
    print("Misp Accuracy: ", acc)


    # for sents in zip(datasets["test-corr"], datasets["test-misp"]):
    #     print(sents[0].tokenized)
    #     print(sents[1].tokenized)
    #     break
    
    # Misspelling Average Embedding [MAE]
    print("")
    print("MAE ...")
    maeModel = MAEClassifier(model)

    test_iter, corr_iter, misp_iter = DATASETS.iter(args, (MAEdatasets["test"], MAEdatasets["test-corr"], MAEdatasets["test-misp"]), device)


    acc, test_loss, predict, labels = MODEL.evaluate(test_iter, maeModel, return_pred=True)
    print("Test Accuracy: ", acc)


    acc, test_loss, predict, labels = MODEL.evaluate(corr_iter, maeModel, return_pred=True)
    print("Corr Accuracy: ", acc)


    acc, test_loss, predict, labels = MODEL.evaluate(misp_iter, maeModel, return_pred=True)
    print("Misp Accuracy: ", acc)
